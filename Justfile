# hsl-rt-analysis — task runner. Run `just` to list the recipes.
#
# Shared conventions across our projects (see the "Task Runner (Justfile)" section of the
# global development guidelines in ~/.claude/CLAUDE.md):
#   just              lists recipes; never has side effects
#
# Recipe names are lowercase kebab-case; anything outside the shared core set is named
# <area>-<verb>. Domain commands live in the `project` group.

set positional-arguments := true

# HFP_ARCHIVE_REMOTE (the storage box user@host) lives in .env, which is not committed.
set dotenv-load := true
set dotenv-required := false

# Only the maintained package and its tests: exploration/ and docker/ predate the strict rule set.
lint_paths := "src tests"
# The archive converter's tmux session and log, used by the archive-* recipes.
archive_session := "hfp-archive"
archive_log := env("HOME") / "hfp-archive-work/logs/run.log"

[private]
default:
    @just --list --unsorted

# --- setup -------------------------------------------------------------------

# Install/sync the Python virtual environment.
[group('setup')]
install:
    uv sync

# One-time developer setup for a fresh clone: dependencies + git hooks.
[doc('One-time developer setup: dependencies + git hooks.')]
[group('setup')]
setup: install
    uvx prek install

# `hooks` runs the hooks; installing them is `setup`'s job.
[doc('Run all git hooks against all files.')]
[group('setup')]
hooks:
    uvx prek run --all-files

# --- quality -----------------------------------------------------------------

# Run the test suite.
[group('quality')]
test *args:
    uv run pytest "$@"

# What is measured lives in pyproject.toml under [tool.coverage.run].
[doc('Run the test suite with a coverage report.')]
[group('quality')]
cov *args:
    uv run pytest --cov --cov-report=term-missing "$@"

# Lint and format-check (no changes written).
[doc('Lint and format-check (no changes written).')]
[group('quality')]
lint:
    uv run ruff check {{ lint_paths }}
    uv run ruff format --check {{ lint_paths }}

# Auto-fix lint issues and format the code.
[group('quality')]
fmt:
    uv run ruff check --fix {{ lint_paths }}
    uv run ruff format {{ lint_paths }}

# The local pre-push gate.
[group('quality')]
ci: lint test

# --- project -----------------------------------------------------------------

# Any hfp-archive command in the foreground, e.g. `just archive run --days 2026-10-15 --dry-run`.
[doc('Run an hfp-archive command, e.g. just archive run --days 2026-10-15 --dry-run')]
[group('project')]
archive *args:
    uv run hfp-archive "$@"

# Starts the bulk run in a detached tmux session, logging to archive_log. It survives closing the
# terminal or the SSH session. On macOS it runs under caffeinate (-i idle sleep, -s system sleep on AC
# power); a closed laptop lid still sleeps unless the Mac runs clamshell on power, and the run then
# resumes after waking. A Linux server does not sleep, so there it runs as is. Extra arguments go to
# `hfp-archive run`, e.g. `just archive-bg --to 2025-12-31`.
[doc('Start the bulk run in the background (tmux, + caffeinate on macOS), log to ~/hfp-archive-work/logs/run.log')]
[group('project')]
archive-bg *args:
    #!/usr/bin/env bash
    set -euo pipefail
    if tmux has-session -t {{ archive_session }} 2>/dev/null; then
        echo "already running: tmux attach -t {{ archive_session }}"; exit 1
    fi
    : "${HFP_ARCHIVE_REMOTE:?set HFP_ARCHIVE_REMOTE in .env}"
    mkdir -p "$(dirname {{ archive_log }})"
    awake=()
    if command -v caffeinate >/dev/null; then awake=(caffeinate -is); fi
    # A tmux server that is already running (another session) starts the new one with the server's own
    # environment, not this shell's, so PATH (for uv) and the .env settings are passed explicitly. If the
    # run fails, its window stays open with the error, and the exit status always goes to the log.
    tmux new-session -d -s {{ archive_session }} -c "{{ justfile_directory() }}" \
        env PATH="$PATH" HFP_ARCHIVE_REMOTE="$HFP_ARCHIVE_REMOTE" \
        bash -c '"$@"; rc=$?
            echo "$(date +%FT%T) hfp-archive exited with status $rc" >> "{{ archive_log }}"
            if [ "$rc" -ne 0 ]; then echo "exited with status $rc; press Enter to close"; read -r _; fi' \
        archive-bg ${awake[@]+"${awake[@]}"} uv run hfp-archive run --log-file {{ archive_log }} "$@"
    sleep 5
    if ! pgrep -f "bin/hfp-archive run" >/dev/null; then
        echo "the run stopped at once; see the error with: tmux attach -t {{ archive_session }}"; exit 1
    fi
    echo "started; follow with: just archive-log   (or tmux attach -t {{ archive_session }})"

# Follow the bulk run's log.
[group('project')]
archive-log:
    tail -f {{ archive_log }}

# Progress per day, e.g. `just archive-status --all`.
[doc('Progress per day of the archive conversion')]
[group('project')]
archive-status *args:
    uv run hfp-archive status "$@"

# Ask the background run to stop after its current step (SIGTERM to the converter, not to tmux).
[doc('Stop the background run cleanly after its current step')]
[group('project')]
archive-stop:
    # Matches the Python process only, not the uv or caffeinate wrappers around it.
    pkill -TERM -f "bin/hfp-archive run" || echo "not running"
