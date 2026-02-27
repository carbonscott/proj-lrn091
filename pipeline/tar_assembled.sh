#!/usr/bin/env bash
# Tar assembled Zarr stores grouped by experiment+run.
#
# Creates one .tar per run (e.g. cxi101235425_r0100.tar) containing all
# chunk .zarr directories for that run.  A manifest file (tarred_runs.txt)
# tracks which runs have been tarred so re-runs are idempotent — existing
# tars are never overwritten.
#
# Usage:
#   bash exploration/scripts/tar_assembled.sh
#   bash exploration/scripts/tar_assembled.sh --jobs 4
#   bash exploration/scripts/tar_assembled.sh --output-dir /tmp/tarballs
#   bash exploration/scripts/tar_assembled.sh --assembled-dir data/assembled

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ASSEMBLED_DIR="$PROJECT_ROOT/data/assembled"
OUTPUT_DIR="$PROJECT_ROOT/data/assembled_tarballs"
JOBS=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --assembled-dir) ASSEMBLED_DIR="$2"; shift 2 ;;
        --jobs|-j)       JOBS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--jobs N] [--output-dir DIR] [--assembled-dir DIR]"
            echo "  --jobs N         Number of parallel tar workers (default: 1)"
            echo "  --assembled-dir  Directory with .zarr stores (default: data/assembled/)"
            echo "  --output-dir     Directory for .tar files (default: data/assembled_tarballs/)"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

MANIFEST="$OUTPUT_DIR/tarred_runs.txt"

mkdir -p "$OUTPUT_DIR"
touch "$MANIFEST"

echo "Assembled dir: $ASSEMBLED_DIR"
echo "Output dir:    $OUTPUT_DIR"
echo "Manifest:      $MANIFEST"
echo "Jobs:          $JOBS"
echo

# ---------------------------------------------------------------------------
# Worker function: tar a single run prefix
# ---------------------------------------------------------------------------
tar_one_run() {
    local prefix="$1"
    local assembled_dir="$2"
    local output_dir="$3"
    local manifest="$4"

    local tar_file="$output_dir/${prefix}.tar"

    # Collect all chunk dirs for this run
    local chunks=()
    for zarr_dir in "$assembled_dir"/${prefix}.*.zarr; do
        [ -d "$zarr_dir" ] || continue
        chunks+=("$(basename "$zarr_dir")")
    done

    if [ ${#chunks[@]} -eq 0 ]; then
        echo "SKIP (no chunks found): $prefix"
        return
    fi

    echo "Tarring $prefix: ${#chunks[@]} chunk(s) -> $(basename "$tar_file")"

    # Create tar from assembled dir so paths inside tar are relative
    tar cf "$tar_file" -C "$assembled_dir" "${chunks[@]}"

    local tar_size
    tar_size=$(du -sh "$tar_file" | cut -f1)
    echo "  Done: $prefix $tar_size"

    # Record in manifest (single-line append is atomic on POSIX)
    echo "$prefix" >> "$manifest"
}
export -f tar_one_run

# ---------------------------------------------------------------------------
# Collect unique run prefixes and filter out already-tarred ones
# ---------------------------------------------------------------------------
run_prefixes=()
for zarr_dir in "$ASSEMBLED_DIR"/*.zarr; do
    [ -d "$zarr_dir" ] || continue
    basename="$(basename "$zarr_dir")"
    # Strip .NNNN.zarr to get the run prefix
    prefix="${basename%%.*.zarr}"
    run_prefixes+=("$prefix")
done

# Deduplicate and sort
mapfile -t unique_prefixes < <(printf '%s\n' "${run_prefixes[@]}" | sort -u)

echo "Found ${#unique_prefixes[@]} unique run(s)"

# Separate into skip vs. todo lists
todo_prefixes=()
skipped=0
for prefix in "${unique_prefixes[@]}"; do
    if grep -qxF "$prefix" "$MANIFEST" 2>/dev/null; then
        echo "SKIP (already tarred): $prefix"
        skipped=$((skipped + 1))
    else
        todo_prefixes+=("$prefix")
    fi
done

echo "${#todo_prefixes[@]} to tar, $skipped already done"
echo

if [ ${#todo_prefixes[@]} -eq 0 ]; then
    echo "Nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# Dispatch: parallel or sequential
# ---------------------------------------------------------------------------
printf '%s\n' "${todo_prefixes[@]}" | \
    xargs -P "$JOBS" -I {} bash -c \
        'tar_one_run "$@"' _ {} "$ASSEMBLED_DIR" "$OUTPUT_DIR" "$MANIFEST"

tarred=${#todo_prefixes[@]}

echo
echo "Summary: $tarred tarred, $skipped skipped (already in manifest)"
echo "Manifest: $MANIFEST"
