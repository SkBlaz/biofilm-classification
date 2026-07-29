# AGENTS.md

This file is the bot onboarding guide for this repository.

## 1) Purpose

Use this repository to process 3D biofilm images and run feature generation, ranking, learning benchmarks, visualization, and inference pipelines.

## 2) Repository map (quick orientation)

- `src/`: main Python pipeline code
- `src/visualizations/`: visualization modules
- `test_*.py`: unit tests
- `run_tests.py`: test runner
- `run_gui.sh` / `run_gui.bat`: local GUI launchers for Unix-like systems and Windows
- `src/requirements.docker.txt`: runtime Python dependencies used in Docker
- `.github/workflows/`: CI workflows (Docker workflows + Ruff)
- `RUN.md`: usage and Docker execution guide
- `APPROACH.md`: methodology overview

## 3) Local setup and validation commands

Run from repository root:

```bash
python -m pip install -r src/requirements.docker.txt
python -m pip install ruff
PYTHONPATH=src python run_tests.py
python -m ruff check .
python -m ruff format --check .
```

Notes:
- Tests import modules from `src`, so keep `PYTHONPATH=src`.
- CI runs Ruff checks and format checks on `main`/`master` pushes and PRs.

## 4) Change workflow for bots

1. Understand the issue and identify the smallest safe change.
2. Run baseline checks before editing (tests/lint as available).
3. Edit only directly relevant files.
4. Re-run targeted validation after edits.
5. Update this `AGENTS.md` if onboarding/process information changed.

## 5) What to avoid

- Do not change unrelated code or tests.
- Do not commit generated artifacts or large temporary outputs.
- Do not remove existing tests to make CI pass.

## 6) Main-branch bot update rule (required)

Whenever a bot-authored PR is merged to `main` (or `master`), that PR **must** update this `AGENTS.md` with any onboarding/process changes introduced by the PR.

If nothing in onboarding/process changed, add a short entry in the log below saying "Reviewed, no onboarding changes required."

## 7) Bot main-branch touch log

Keep newest entry at the top.

| Date (UTC) | Bot/Agent | PR/Commit | AGENTS.md update summary |
|---|---|---|---|
| 2026-07-29 | Codex | current change | Sequential GUI workflow, complete external-table contract, imaging-position replication, Windows diagnostics, and generated-feature validation documented. |
| 2026-07-21 | Codex | current change | Unix GUI launcher now diagnoses sudo misuse and Docker socket access. |
| 2026-07-21 | Codex | current change | GUI preflight validation, grouped replication-aware benchmarking, resumable learner runs, organized reports, demo workflow, citation metadata, and UTF-8 Windows launcher documented. |
| 2026-07-10 | Codex | current change | Windows GUI launcher and icon shortcut workflow documented. |
| 2026-05-15 | @copilot | current PR | Initial detailed onboarding guide added. |
