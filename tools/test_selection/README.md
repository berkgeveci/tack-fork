# Test selection prototype (#369)

Which tests does a pull request's change reach? A prototype that records, per
test, the repository files each test used, and replays past pull requests and
failing CI runs against that map. It is a measurement, not a CI change:
nothing here runs in CI. The results of 2026-11-05 are in #369.

## Record a map

From the repository root, on a CPU-only environment (no `metal` extra: the run
uses xdist):

```bash
OUT=results/test_selection/tests           # or .../files with depmap_plugin
mkdir -p $OUT
env DEPMAP_ROOT=$PWD DEPMAP_OUT=$PWD/$OUT \
    PYTHONPATH=$PWD/tools/test_selection:$PWD/packages/tack-core/src:$PWD/packages/tack-vis/src \
    uv run pytest -q -p no:cacheprovider -p depmap_plugin_tests -n 8 --dist loadfile
```

- `depmap_plugin_tests` records per test (setup and teardown are credited to
  every test of the file); `depmap_plugin` records per test file.
- `depmap_core.py` holds the recorder:
  - `sys.monitoring` `PY_START` events for functions, each location disabled
    after its first hit and re-enabled at the next test;
  - hooks on `Kernel.__call__`, `Kernel._lower` and `CallBindings.resolve`,
    which credit a kernel call with the kernel's file and every file lowering
    inlined into it.
- `sitecustomize.py` records subprocesses into the test that started them. It
  shadows the interpreter's own `sitecustomize`, then runs it.
- About 6.5 minutes on 8 workers of an M1 Max, no slower than a plain run.

## Replay

```bash
python3 tools/test_selection/replay_tests.py   # per test
python3 tools/test_selection/replay_files.py   # per test file
```

- The replays read the maps from `results/test_selection/{tests,files}`, or
  from the directory `$DEPMAP_MAPS` names.
- They use `tools/ci_changes.py` (the path filter, #368) and need `origin/main`
  and the pull requests' commits in the clone.
- `history/` holds the merged PRs and failing runs from 2026-10-14 to
  2026-11-05; `history/fetch.py SINCE` refreshes it.
