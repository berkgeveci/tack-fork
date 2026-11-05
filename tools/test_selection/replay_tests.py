# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay merged PRs and failing CI runs against the per-test dependency map."""
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
MAP = os.path.join(os.environ.get("DEPMAP_MAPS", os.path.join(WT, "results", "test_selection")), "tests")
sys.path.insert(0, os.path.dirname(HERE))           # tools/, for ci_changes
import ci_changes

raw = collections.defaultdict(set)
for f in glob.glob(os.path.join(MAP, "pytest-*.json")) + glob.glob(os.path.join(MAP, "sub-*.json")):
    for key, used in json.load(open(f)).items():
        raw[key] |= set(used)
local = {}
for f in glob.glob(os.path.join(MAP, "durations-*.json")):
    local.update(json.load(open(f)))

def file_of(nodeid):
    return nodeid.split("::")[0]

tests_in = collections.defaultdict(set)
for nodeid in local:
    tests_in[file_of(nodeid)].add(nodeid)

deps = collections.defaultdict(set)          # node id -> files
for key, used in raw.items():
    if key.startswith("setup::"):
        for t in tests_in[key[len("setup::"):]]:
            deps[t] |= used
    elif "::" in key:
        deps[key] |= used
users = collections.defaultdict(set)
for t, used in deps.items():
    for u in used:
        users[u].add(t)

ci = json.load(open(os.path.join(WT, "test-durations.json")))
TOTAL = sum(ci.values())
weight = {}
for f, ts in tests_in.items():
    lt = sum(local[t] for t in ts)
    for t in ts:
        weight[t] = ci.get(f, 0.0) * (local[t] / lt if lt else 1 / len(ts))
FULL = {"pyproject.toml", "uv.lock", "conftest.py", "test-durations.json"}


def is_test(path):
    return "/tests/" in path and os.path.basename(path).startswith("test_") and path.endswith(".py")


def select(files, added=()):
    tests = set()
    for f in files:
        if "code" not in ci_changes.groups_of(f):
            continue
        if f in FULL or f.startswith(".github/") or f.endswith("conftest.py"):
            return None, f"build/CI file {f}"
        if is_test(f):
            tests |= tests_in.get(f, set())
            continue
        if f in users:
            tests |= users[f]
        elif f in added or not re.match(r"packages/[^/]+/src/", f):
            continue
        else:
            return None, f"unmapped {f}"
    return tests, None


def cost(tests):
    return 1.0 if tests is None else sum(weight.get(t, 0.0) for t in tests) / TOTAL


def git(*a):
    return subprocess.run(["git", "-C", WT, *a], capture_output=True, text=True).stdout


def main():
    print(f"map: {len(deps)} tests in {len(tests_in)} files, {len(users)} files used")
    prs = json.load(open(os.path.join(HISTORY, "merged.json")))
    rows = []
    for p in prs:
        m = p["mergeCommit"]["oid"]
        tests, why = select(p["files"], set(git("diff", "--name-only", "--diff-filter=A", f"{m}^1", m).split()))
        rows.append((p["number"], cost(tests), why))
    costs = [r[1] for r in rows]
    partial = [c for (_, c, why) in rows if not why and c > 0]
    print(f"\n{len(rows)} merged PRs since 2026-10-14")
    print(f"  no test job: {sum(1 for c in costs if c == 0)}")
    print(f"  full run: {sum(1 for r in rows if r[2])}")
    print(f"  partial: {len(partial)}; median {statistics.median(partial):.0%}, mean {statistics.mean(partial):.0%}")
    print(f"  overall test time vs today: {statistics.mean(costs):.0%}")
    print("  partial runs by tenth of suite:",
          dict(sorted(collections.Counter(min(int(c * 10), 9) for c in partial).items())))

    print("\nFailing CI runs:")
    for f in json.load(open(os.path.join(HISTORY, "failures.json"))):
        lines = open(os.path.join(HISTORY, "logs", f"{f['run']}.txt")).read().splitlines()
        failed = sorted({m.group(1) for l in lines for m in [re.search(r"(?:FAILED|ERROR) (packages/.+?)(?: - |$)", l)] if m})
        main_then = git("rev-list", "-1", "--first-parent", f"--before={f['created']}", "origin/main").strip()
        base = git("merge-base", main_then, f["sha"]).strip()
        tests, why = select(f["files"], set(git("diff", "--name-only", "--diff-filter=A", base, f["sha"]).split()))
        if tests is None:
            print(f"  {f['run']} {f['branch']:<30} full run ({why})")
            continue
        known = [t for t in failed if t in local]
        caught = [t for t in known if t in tests]
        unknown = [t for t in failed if t not in local]
        files_ok = all(any(file_of(s) == file_of(t) for s in tests) for t in unknown)
        print(f"  {f['run']} {f['branch']:<30} {cost(tests):4.0%}  failing tests caught {len(caught)}/{len(known)}"
              + (f"; {len(unknown)} no longer exist, their files {'selected' if files_ok else 'NOT selected'}" if unknown else ""))
        for t in known:
            if t not in tests:
                print("      MISSED", t)


main()
