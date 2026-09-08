#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"
OUTPUT_DIR="$ROOT_DIR/output"

MODE=fresh
INSTALL=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: ./run_all.sh [--fresh|--resume] [--install] [--dry-run]

Run empirical and simulation notebooks sequentially, then results-only plotters.
Each notebook parallelizes its own independent seed/model jobs.

  --fresh    Clear output/, then recompute every seed/model result (default).
  --resume   Reuse valid checkpoints and compute only missing work.
  --install  Install requirements before running.
  --dry-run  Print the notebooks and settings without executing them.
EOF
}

while (($#)); do
    case "$1" in
        --fresh)
            MODE=fresh
            ;;
        --resume)
            MODE=resume
            ;;
        --install)
            INSTALL=1
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

export IMV_N_JOBS=${IMV_N_JOBS:-15}
export IMV_TORCH_JOBS=${IMV_TORCH_JOBS:-6}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/imv-mplconfig-${UID}}
export IMV_FIGURE_DIR="${IMV_FIGURE_DIR:-$OUTPUT_DIR/figures}"
export PYTHONUNBUFFERED=1

for setting in IMV_N_JOBS IMV_TORCH_JOBS; do
    value=${!setting}
    if [[ ! $value =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer, got %q\n' "$setting" "$value" >&2
        exit 2
    fi
done

if [[ $MODE == fresh ]]; then
    export IMV_FORCE_RECOMPUTE=1
    resolved_output_dir=$(realpath -m "$OUTPUT_DIR")
    resolved_figure_dir=$(realpath -m "$IMV_FIGURE_DIR")
    if [[ $resolved_figure_dir/ != "$resolved_output_dir/"* ]]; then
        printf 'Fresh mode requires IMV_FIGURE_DIR to be under %s, got %s\n' \
            "$resolved_output_dir" "$resolved_figure_dir" >&2
        exit 2
    fi
    OUTPUT_DIR=$resolved_output_dir
    export IMV_FIGURE_DIR=$resolved_figure_dir
else
    export IMV_FORCE_RECOMPUTE=0
fi

mapfile -d '' NOTEBOOKS < <(
    find src/empirical src/simulations -type f -name '*.ipynb' \
        ! -path 'src/empirical/plotter/*' ! -path '*/.ipynb_checkpoints/*' -print0 | sort -z
)
# Results-only plotters must follow every experiment producer.
while IFS= read -r -d '' notebook; do
    NOTEBOOKS+=("$notebook")
done < <(find src/empirical/plotter -type f -name '*.ipynb' ! -path '*/.ipynb_checkpoints/*' -print0 | sort -z)
if ((${#NOTEBOOKS[@]} == 0)); then
    printf 'No notebooks found under %s/src\n' "$ROOT_DIR" >&2
    exit 1
fi

printf 'Mode: %s\n' "$MODE"
if [[ $MODE == fresh ]]; then
    printf 'Checkpoints: ignored and atomically replaced as jobs finish\n'
    printf 'Outputs: %s will be emptied before execution\n' "$OUTPUT_DIR"
else
    printf 'Checkpoints: valid completed jobs will be restored\n'
    printf 'Outputs: existing files will be preserved\n'
fi
printf 'Workers: IMV_N_JOBS=%s, IMV_TORCH_JOBS=%s\n' \
    "$IMV_N_JOBS" "$IMV_TORCH_JOBS"
printf 'Figures: %s\n' "$IMV_FIGURE_DIR"
printf 'Notebooks: %d (run sequentially; jobs inside each run in parallel)\n' \
    "${#NOTEBOOKS[@]}"

if ((DRY_RUN)); then
    for index in "${!NOTEBOOKS[@]}"; do
        printf '[%d/%d] %s\n' \
            "$((index + 1))" "${#NOTEBOOKS[@]}" "${NOTEBOOKS[$index]}"
    done
    exit 0
fi

command -v flock >/dev/null 2>&1 || {
    printf 'flock is required to prevent overlapping notebook runs.\n' >&2
    exit 1
}
command -v setsid >/dev/null 2>&1 || {
    printf 'setsid is required to clean up notebook kernels on interruption.\n' >&2
    exit 1
}

LOCK_ROOT=${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}}
LOCK_FILE="$LOCK_ROOT/imv-ml-run-all-${UID}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf 'Another run_all.sh invocation is already active.\n' >&2
    exit 1
fi

mkdir -p "$MPLCONFIGDIR"

if ((INSTALL)); then
    printf '\nInstalling requirements...\n'
    python -m pip install --disable-pip-version-check --quiet -r requirements.txt
fi

if [[ $MODE == fresh ]]; then
    printf '\nClearing generated outputs under %s...\n' "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    find "$OUTPUT_DIR" -mindepth 1 -delete
fi
mkdir -p "$IMV_FIGURE_DIR"

format_duration() {
    local total=$1
    printf '%dm %02ds' "$((total / 60))" "$((total % 60))"
}

stop_process_group() {
    local pgid=$1
    local attempt

    if ! kill -0 -- "-$pgid" 2>/dev/null; then
        return
    fi
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for ((attempt = 0; attempt < 50; attempt++)); do
        if ! kill -0 -- "-$pgid" 2>/dev/null; then
            return
        fi
        sleep 0.1
    done
    kill -KILL -- "-$pgid" 2>/dev/null || true
}

current_pid=
cleanup() {
    local status=$?
    trap - EXIT
    if [[ -n ${current_pid:-} ]]; then
        stop_process_group "$current_pid"
        wait "$current_pid" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_started=$(date +%s)
for index in "${!NOTEBOOKS[@]}"; do
    notebook=${NOTEBOOKS[$index]}
    notebook_started=$(date +%s)
    printf '\n[%d/%d] Running %s\n' \
        "$((index + 1))" "${#NOTEBOOKS[@]}" "$notebook"

    setsid jupyter nbconvert \
        --execute \
        --to notebook \
        --inplace \
        --ExecutePreprocessor.timeout=-1 \
        "$notebook" &
    current_pid=$!

    if wait "$current_pid"; then
        status=0
    else
        status=$?
    fi
    completed_pid=$current_pid

    if ((status != 0)); then
        printf '[%d/%d] Failed %s (exit %d)\n' \
            "$((index + 1))" "${#NOTEBOOKS[@]}" "$notebook" "$status" >&2
        exit "$status"
    fi

    current_pid=
    stop_process_group "$completed_pid"
    notebook_elapsed=$(($(date +%s) - notebook_started))
    printf '[%d/%d] Completed %s in %s\n' \
        "$((index + 1))" "${#NOTEBOOKS[@]}" "$notebook" \
        "$(format_duration "$notebook_elapsed")"
done

run_elapsed=$(($(date +%s) - run_started))
printf '\nAll %d notebooks completed in %s.\n' \
    "${#NOTEBOOKS[@]}" "$(format_duration "$run_elapsed")"
