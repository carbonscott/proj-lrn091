"""Explore HDF5 structure across experiment symlinks to find image datasets.

For each symlink in data/, finds a sample .cxi or .h5 file, walks the HDF5 key
tree, and identifies image-like datasets (2D+ arrays with spatial dims > 100).

Usage:
    uv run --with h5py exploration/scripts/explore_hdf5.py
"""

import json
import os
import sys
from pathlib import Path

import h5py

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

MAX_DEPTH = 5  # Max HDF5 group nesting to traverse
MAX_FILES_TO_LIST = 200  # Cap on file listing per symlink to avoid slow dirs


def find_sample_files(symlink_path, max_files=3):
    """Find a few sample .cxi or .h5 files under the symlink target."""
    extensions = {".cxi", ".h5", ".hdf5"}
    found = []

    # Walk breadth-first, but limit depth and file count
    for dirpath, dirnames, filenames in os.walk(symlink_path):
        depth = dirpath.replace(str(symlink_path), "").count(os.sep)
        if depth > 4:
            dirnames.clear()
            continue

        for fname in filenames:
            if Path(fname).suffix.lower() in extensions:
                found.append(Path(dirpath) / fname)
                if len(found) >= max_files:
                    return found

        if len(found) >= max_files:
            break

    return found


def walk_hdf5(group, prefix="", depth=0, results=None):
    """Recursively walk HDF5 groups and collect dataset metadata."""
    if results is None:
        results = []
    if depth > MAX_DEPTH:
        return results

    for key in group:
        full_key = f"{prefix}/{key}"
        try:
            item = group[key]
        except Exception:
            continue

        if isinstance(item, h5py.Dataset):
            results.append({
                "key": full_key,
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "nbytes": item.nbytes,
            })
        elif isinstance(item, h5py.Group):
            walk_hdf5(item, full_key, depth + 1, results)

    return results


def is_image_like(ds_info):
    """Check if a dataset looks like it contains image data."""
    shape = ds_info["shape"]
    dtype = ds_info["dtype"]

    # Skip non-numeric types
    if "str" in dtype or "object" in dtype or "bool" in dtype:
        return False

    ndim = len(shape)

    # 2D: single image (H, W)
    if ndim == 2 and shape[0] > 100 and shape[1] > 100:
        return True

    # 3D: stack of images (N, H, W) or multi-panel (panels, H, W)
    if ndim == 3 and shape[1] > 100 and shape[2] > 100:
        return True

    # 4D: stack of multi-panel images (N, panels, H, W)
    if ndim == 4 and shape[2] > 100 and shape[3] > 100:
        return True

    return False


def explore_symlink(symlink_name, symlink_path):
    """Explore one symlink and return a summary dict."""
    print(f"\n{'='*60}")
    print(f"  {symlink_name}")
    print(f"  -> {symlink_path}")
    print(f"{'='*60}")

    result = {
        "symlink": symlink_name,
        "target": str(symlink_path),
        "files_explored": [],
        "image_datasets": [],
    }

    sample_files = find_sample_files(symlink_path)
    if not sample_files:
        print("  No .cxi/.h5/.hdf5 files found.")
        return result

    for fpath in sample_files:
        rel = fpath.relative_to(symlink_path)
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"\n  File: {rel} ({size_mb:.1f} MB)")

        file_info = {
            "path": str(fpath),
            "relative": str(rel),
            "size_mb": round(size_mb, 1),
            "datasets": [],
            "image_keys": [],
        }

        try:
            with h5py.File(fpath, "r") as f:
                datasets = walk_hdf5(f)
                file_info["datasets"] = datasets

                for ds in datasets:
                    shape_str = "x".join(str(s) for s in ds["shape"])
                    marker = ""
                    if is_image_like(ds):
                        marker = "  ** IMAGE **"
                        file_info["image_keys"].append(ds["key"])
                    print(f"    {ds['key']:55s}  {shape_str:25s}  {ds['dtype']:10s}{marker}")

                if file_info["image_keys"]:
                    print(f"\n  -> Image keys: {file_info['image_keys']}")
                else:
                    print(f"\n  -> No image-like datasets found in this file.")

        except PermissionError:
            print(f"    Permission denied.")
            file_info["error"] = "permission_denied"
        except Exception as e:
            print(f"    Error: {e}")
            file_info["error"] = str(e)

        result["files_explored"].append(file_info)

        # Collect unique image keys across files
        for k in file_info["image_keys"]:
            if k not in result["image_datasets"]:
                result["image_datasets"].append(k)

    return result


def main():
    print("HDF5 Structure Explorer")
    print(f"Data directory: {DATA_DIR}")

    # Find all symlinks in data/
    symlinks = sorted([
        p for p in DATA_DIR.iterdir()
        if p.is_symlink() and "--" in p.name
    ])

    print(f"Found {len(symlinks)} experiment symlinks.\n")

    all_results = []
    for sl in symlinks:
        target = Path(os.readlink(sl))
        if not target.is_absolute():
            target = (sl.parent / target).resolve()

        result = explore_symlink(sl.name, target)
        all_results.append(result)

    # Write JSON summary
    summary_path = DATA_DIR / "hdf5_structure_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nJSON summary written to: {summary_path}")

    # Print a concise summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Symlink':<40s} {'Image Keys'}")
    print(f"{'-'*40} {'-'*40}")
    for r in all_results:
        keys = ", ".join(r["image_datasets"]) if r["image_datasets"] else "(none found)"
        print(f"{r['symlink']:<40s} {keys}")


if __name__ == "__main__":
    main()
