# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay merged PRs and failing CI runs against the per-test-file dependency map."""
import collections
import glob
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(os.path.dirname(HERE))          # the repository (needs origin/main)
HISTORY = os.path.join(HERE, "history")
MAP = os.path.join(os.environ.get("DEPMAP_MAPS", os.path.join(WT, "results", "test_selection")), "files")
sys.path.insert(0, os.path.dirname(HERE))           # tools/, for ci_changes
import ci_changes

# The map: test file -> files it used (pytest workers and their subprocesses).
deps = collections.defaultdict(set)
for f in glob.glob(os.path.join(MAP, "*.json")):
    for test, used in json.load(open(f)).items():
        deps[test] |= set(used)
users = collections.defaultdict(set)          # file -> test files that used it
for test, used in deps.items():
    for u in used:
        users[u].add(test)

durations = json.load(open(os.path.join(WT, "test-durations.json")))
TOTAL = sum(durations.values())
FULL = {"pyproject.toml", "uv.lock", "conftest.py", "test-durations.json"}


def is_test(path):
    return "/tests/" in path and os.path.basename(path).startswith("test_") and path.endswith(".py")


def select(files, added=()):
    """(tests, reason for a full run or None)."""
    tests = set()
    for f in files:
        if "code" not in ci_changes.groups_of(f):
            continue                       # docs, examples, tools: no test job
        if f in FULL or f.startswith(".github/") or f.endswith("conftest.py"):
            return None, f"build/CI file {f}"
        if is_test(f):
            if f in durations or f in deps:
                tests.add(f)
            continue                       # a deleted test file
        if f in users:
            tests |= users[f]
        elif f in added:
            continue                       # new: reached through the changed files that use it
        elif not re.match(r"packages/[^/]+/src/", f):
            continue                       # a script or data no test runs
        else:
            return None, f"unmapped {f}"
    return tests, None


def cost(tests):
    return 1.0 if tests is None else sum(durations.get(t, 0.0) for t in tests) / TOTAL


def added_files(base, head):
    out = subprocess.run(["git", "-C", WT, "diff", "--name-only", "--diff-filter=A", base, head],
                         capture_output=True, text=True).stdout.split()
    return set(out)


def main():
    print(f"map: {len(deps)} test files, {len(users)} files used; suite {TOTAL / 60:.0f} min of test time")
    unseen = [t for t in durations if t not in deps]
    print(f"test files in test-durations.json with no map entry: {len(unseen)} {unseen[:5]}")

    prs = json.load(open(os.path.join(HISTORY, "merged.json")))
    rows, reasons = [], collections.Counter()
    for p in prs:
        m = p["mergeCommit"]["oid"]
        tests, why = select(p["files"], added_files(f"{m}^1", m))
        c = cost(tests)
        rows.append((p["number"], c, why, len(tests or ())))
        if why:
            reasons[why.split(" ")[0] + " " + (why.split(" ")[1] if why.startswith("build") else "")] += 1
    costs = [r[1] for r in rows]
    none = sum(1 for c in costs if c == 0)
    full = sum(1 for r in rows if r[2])
    print(f"\n{len(rows)} merged PRs since 2026-10-14")
    print(f"  no test job (docs/examples/tools only): {none}")
    print(f"  full run: {full}  ({dict(reasons)})")
    partial = [c for r, c in zip(rows, costs) if not r[2] and c > 0]
    print(f"  partial: {len(partial)}; median {statistics.median(partial):.0%} of suite time, "
          f"mean {statistics.mean(partial):.0%}" if partial else "  partial: 0")
    print(f"  overall test time vs today: {statistics.mean(costs):.0%}")
    buckets = collections.Counter(min(int(c * 10), 9) for c in costs if 0 < c < 1)
    print("  partial runs by tenth of suite:", dict(sorted(buckets.items())))
    why_full = collections.Counter(r[2] for r in rows if r[2])
    print("  commonest full-run causes:", why_full.most_common(12))

    print("\nFailing CI runs: was every failing test file selected?")
    for f in json.load(open(os.path.join(HISTORY, "failures.json"))):
        main_then = subprocess.run(["git", "-C", WT, "rev-list", "-1", "--first-parent",
                                    f"--before={f['created']}", "origin/main"],
                                   capture_output=True, text=True).stdout.strip()
        base = subprocess.run(["git", "-C", WT, "merge-base", main_then, f["sha"]],
                              capture_output=True, text=True).stdout.strip()
        tests, why = select(f["files"], added_files(base, f["sha"]))
        missed = [] if tests is None else [t for t in f["fails"] if t not in tests]
        verdict = "full run" if tests is None else ("MISSED " + ", ".join(missed) if missed else "caught")
        print(f"  {f['run']} {f['branch']:<32} {verdict:<40} {'' if tests is None else f'{cost(tests):.0%}'}"
              f"  {why or ''}")


main()
