#!/bin/bash
# Create symlinks under data/ for easy access to experiment HDF5 data.
#
# Usage:
#   bash create_symlinks.sh --config symlinks.yml
#   bash create_symlinks.sh --config symlinks.yml --data-dir /path/to/data
#   bash create_symlinks.sh --config symlinks.yml --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR=""
CONFIG=""
DRY_RUN=false

usage() {
    echo "Usage: $0 --config <file> [--data-dir <path>] [--dry-run]"
    echo ""
    echo "Options:"
    echo "  --config    YAML file with symlink_name: source_path entries (required)"
    echo "  --data-dir  Target directory for symlinks (default: ../data relative to script)"
    echo "  --dry-run   Preview symlinks without creating them"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        -h|--help)  usage ;;
        *)          echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required."
    usage
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Default data dir: one level up from script, then into data/
if [[ -z "$DATA_DIR" ]]; then
    DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/data"
fi

mkdir -p "${DATA_DIR}/sample-images"

# Parse YAML (simple key: value format, no nested structures).
# Skips comments and blank lines.
count=0
while IFS= read -r line; do
    # Skip comments and blank lines
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// /}" ]] && continue

    # Extract key and value (name: /path/to/source)
    name="$(echo "$line" | sed 's/:.*//' | xargs)"
    source="$(echo "$line" | sed 's/^[^:]*://' | xargs)"

    if [[ -z "$name" || -z "$source" ]]; then
        continue
    fi

    if $DRY_RUN; then
        echo "[dry-run] ln -sfn $source -> ${DATA_DIR}/${name}"
    else
        if [[ ! -e "$source" ]]; then
            echo "Warning: source does not exist: $source (skipping $name)"
            continue
        fi
        ln -sfn "$source" "${DATA_DIR}/${name}"
    fi
    count=$((count + 1))
done < "$CONFIG"

if $DRY_RUN; then
    echo ""
    echo "[dry-run] Would create $count symlinks in ${DATA_DIR}/"
else
    echo "Created $count symlinks in ${DATA_DIR}/"
    echo "Created sample-images directory at ${DATA_DIR}/sample-images/"
fi
