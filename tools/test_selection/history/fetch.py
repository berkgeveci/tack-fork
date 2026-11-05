# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Refresh the history the replays read: merged PRs and failing PR runs of ci.yml.

    python3 tools/test_selection/history/fetch.py 2026-10-14

Needs `gh` (logged in) and a clone with origin/main and the PRs' commits."""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "Kitware/tack"


def run(*a):
    return subprocess.run(a, capture_output=True, text=True, check=True).stdout


def git(*a):
    return run("git", *a).strip()


def main(since):
    prs = json.loads(run("gh", "pr", "list", "--repo", REPO, "--state", "merged", "--limit", "1000",
                         "--search", f"merged:>={since}", "--json", "number,title,mergedAt,mergeCommit,author"))
    for p in prs:
        m = p["mergeCommit"]["oid"]
        p["files"] = git("diff", "--name-only", f"{m}^1", m).split()
    json.dump(prs, open(os.path.join(HERE, "merged.json"), "w"), indent=1)

    runs = json.loads(run("gh", "run", "list", "--repo", REPO, "--workflow", "ci.yml", "--event", "pull_request",
                          "--status", "failure", "--limit", "1000", "--created", f">={since}",
                          "--json", "databaseId,headSha,headBranch,createdAt,displayTitle"))
    json.dump(runs, open(os.path.join(HERE, "failed_runs.json"), "w"), indent=1)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    failures = []
    for r in runs:
        log = subprocess.run(["gh", "run", "view", str(r["databaseId"]), "--repo", REPO, "--log-failed"],
                             capture_output=True, text=True).stdout
        lines = sorted({re.sub(r"^[^\t]*\t[^\t]*\t\S+ ", "", l) for l in log.splitlines()
                        if re.search(r"(FAILED|ERROR) packages/|##\[error\]", l)})
        open(os.path.join(HERE, "logs", f"{r['databaseId']}.txt"), "w").write("\n".join(lines) + "\n")
        fails = sorted({m.group(1) for l in lines for m in [re.search(r"(?:FAILED|ERROR) (packages/[^ :]+\.py)", l)] if m})
        if not fails:
            continue
        if subprocess.run(["git", "cat-file", "-e", r["headSha"] + "^{commit}"]).returncode:
            subprocess.run(["git", "fetch", "-q", f"https://github.com/{REPO}.git", r["headSha"]])
        main_then = git("rev-list", "-1", "--first-parent", f"--before={r['createdAt']}", "origin/main")
        base = git("merge-base", main_then, r["headSha"])
        failures.append({"run": r["databaseId"], "sha": r["headSha"], "branch": r["headBranch"],
                             "title": r["displayTitle"], "created": r["createdAt"], "fails": fails,
                             "files": git("diff", "--name-only", base, r["headSha"]).splitlines()})
    json.dump(failures, open(os.path.join(HERE, "failures.json"), "w"), indent=1)
    print(f"{len(prs)} merged PRs, {len(runs)} failed runs, {len(failures)} with named tests")


if __name__ == "__main__":
    main(sys.argv[1])
