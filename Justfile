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
archive_log_dir := env("HOME") / "hfp-archive-work/logs"
archive_log := archive_log_dir / "run.log"

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

# Copies the converted days to the Azure Blob container in a detached tmux session, next to a running
# conversion (it has its own lock). The SAS URL and prefixes come from .env, which hfp-archive reads
# itself, so the token never appears on a command line. Extra arguments go to `hfp-archive replicate`,
# e.g. `just archive-replicate-bg --to 2025-12-31`.
[doc('Copy converted days to Azure in the background (tmux), log to ~/hfp-archive-work/logs/replicate.log')]
[group('project')]
archive-replicate-bg *args:
    #!/usr/bin/env bash
    set -euo pipefail
    session=hfp-replicate log="{{ archive_log_dir }}/replicate.log"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "already running: tmux attach -t $session"; exit 1
    fi
    mkdir -p "{{ archive_log_dir }}"
    tmux new-session -d -s "$session" -c "{{ justfile_directory() }}" env PATH="$PATH" \
        bash -c '"$@"; rc=$?
            echo "$(date +%FT%T) hfp-archive replicate exited with status $rc" >> "'"$log"'"
            if [ "$rc" -ne 0 ]; then echo "exited with status $rc; press Enter to close"; read -r _; fi' \
        archive-replicate-bg uv run hfp-archive replicate --log-file "$log" "$@"
    echo "started; follow with: tail -f $log   (or tmux attach -t $session)"

# One daily pass: convert the days that have passed the cutoff, then copy every finished day to Azure.
# The systemd timer of archive-timer-install runs this. Both steps run even if the first one fails.
[doc('Daily pass: convert new days, then copy finished days to Azure')]
[group('project')]
archive-daily:
    #!/usr/bin/env bash
    set -uo pipefail
    log="{{ archive_log_dir }}/daily.log"
    uv run hfp-archive run --log-file "$log"; run_rc=$?
    uv run hfp-archive replicate --log-file "$log"; rep_rc=$?
    echo "$(date +%FT%T) daily pass: run exited $run_rc, replicate exited $rep_rc" >> "$log"
    exit $(( run_rc || rep_rc ))

# Installs a systemd user timer (Linux) that runs `just archive-daily` every day at 06:30 local time;
# Persistent=true catches up a run missed while the machine was off. For it to run while you are
# logged out, enable lingering once: `loginctl enable-linger "$USER"`.
[doc('Install the daily systemd user timer for archive-daily (Linux)')]
[group('project')]
archive-timer-install:
    #!/usr/bin/env bash
    set -euo pipefail
    units="$HOME/.config/systemd/user"
    mkdir -p "$units"
    printf '%s\n' "[Unit]" "Description=HFP archive daily conversion and Azure copy" \
        "[Service]" "Type=oneshot" "WorkingDirectory={{ justfile_directory() }}" \
        "Environment=PATH=$PATH" "ExecStart=$(command -v just) archive-daily" > "$units/hfp-archive-daily.service"
    printf '%s\n' "[Unit]" "Description=Run the HFP archive daily pass" \
        "[Timer]" "OnCalendar=*-*-* 06:30" "Persistent=true" \
        "[Install]" "WantedBy=timers.target" > "$units/hfp-archive-daily.timer"
    systemctl --user daemon-reload
    systemctl --user enable --now hfp-archive-daily.timer
    systemctl --user list-timers hfp-archive-daily.timer
    loginctl show-user "$USER" -p Linger

# When the daily timer runs next and how its last runs went.
[doc('Show the daily timer and its recent runs (Linux)')]
[group('project')]
archive-timer-status:
    systemctl --user list-timers hfp-archive-daily.timer
    journalctl --user -u hfp-archive-daily.service -n 20 --no-pager
    tail -n 5 "{{ archive_log_dir }}/daily.log"

# Progress of the Azure copy: copied, failed and not yet copied days.
[doc('Progress of the Azure copy')]
[group('project')]
archive-replica-status:
    uv run hfp-archive replica-status

# Runs exploration/sirius_benchmark.py in its own environment (DuckDB 2.0, independent of the project's
# pinned DuckDB). See docs/sirius-benchmark.md, e.g. `just sirius-bench --data ~/sirius-bench/data`.
[doc('Benchmark HFP queries on CPU vs the Sirius GPU extension (Linux + NVIDIA)')]
[group('project')]
sirius-bench *args:
    uv run --script exploration/sirius_benchmark.py "$@"

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
