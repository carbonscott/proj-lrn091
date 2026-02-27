"""Build a JSON manifest of all HDF5 data files for model training.

Walks each experiment symlink in data/, opens every .cxi/.h5 file, reads the
image dataset shape, and writes a structured manifest to data/manifest.json.

The manifest maps each file to its detector type and panel configuration,
enabling the PyTorch DataLoader to efficiently index (file, frame, panel)
triples without re-scanning the filesystem.

Usage:
    uv run --with h5py python build_manifest.py
    uv run --with h5py python build_manifest.py --data-dir /path/to/data --output manifest.json
    uv run --with h5py python build_manifest.py --detector-map experiment_detector_map.yml
    uv run --with h5py python build_manifest.py --experiments cxilw5019--psocake,mfxp22421--cheetah-hdf5
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import h5py

IMAGE_KEY = "/entry_1/data_1/data"

# Detector configurations: raw image shape -> panel layout
# Panel splits are along the H (vertical) dimension.
DETECTOR_CONFIG = {
    "jungfrau_4m": {
        "raw_shape": [4096, 1024],
        "num_panels": 8,
        "panel_h": 512,
        "panel_w": 1024,
    },
    "epix10k_2m": {
        "raw_shape": [5632, 384],
        "num_panels": 16,
        "panel_h": 352,
        "panel_w": 384,
    },
    "jungfrau_16m": {
        "raw_shape": [16384, 1024],
        "num_panels": 32,
        "panel_h": 512,
        "panel_w": 1024,
    },
    "assembled": {
        "raw_shape": [1920, 1920],
        "num_panels": 1,
        "panel_h": 1920,
        "panel_w": 1920,
    },
}

# Default experiment -> detector mapping.
# Can be overridden with --detector-map flag pointing to a YAML file.
_DEFAULT_EXPERIMENT_DETECTOR = {
    "cxi101235425--cheetah-hdf5": "jungfrau_4m",
    "cxil1005322--cheetah-hdf5": "jungfrau_4m",
    "cxil1015922--cheetah-hdf5": "jungfrau_4m",
    "cxilw5019--psocake": "jungfrau_4m",
    "mfx100903824--results-cxi": "epix10k_2m",
    "mfx101211025--cheetah-hdf5": "jungfrau_16m",
    "mfxp22421--cheetah-hdf5": "epix10k_2m",
    "mfxx49820--lute-drcomp-f32": "epix10k_2m",
    "prjcwang31--userdata-cheetah": "epix10k_2m",
    "prjcwang31--userdata-psocake": "assembled",
}


def load_detector_map(path):
    """Load experiment->detector mapping from a simple YAML file (key: value per line)."""
    mapping = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, detector = line.partition(":")
            name = name.strip()
            detector = detector.strip()
            if name and detector:
                mapping[name] = detector
    return mapping


def find_hdf5_files(symlink_path):
    """Find all .cxi/.h5/.hdf5 files under a symlink target."""
    extensions = {".cxi", ".h5", ".hdf5"}
    found = []

    for dirpath, dirnames, filenames in os.walk(symlink_path):
        depth = dirpath.replace(str(symlink_path), "").count(os.sep)
        if depth > 4:
            dirnames.clear()
            continue

        for fname in sorted(filenames):
            if Path(fname).suffix.lower() in extensions:
                fpath = Path(dirpath) / fname
                # Skip very small files (masks, metadata)
                try:
                    if fpath.stat().st_size < 1_000_000:  # < 1 MB
                        continue
                except OSError:
                    continue
                found.append(fpath)

    return found


def inspect_file(fpath, image_key=IMAGE_KEY):
    """Open an HDF5 file and return metadata about the image dataset."""
    try:
        with h5py.File(fpath, "r") as f:
            if image_key not in f:
                return None

            ds = f[image_key]
            shape = list(ds.shape)
            dtype = str(ds.dtype)

            # Determine number of frames
            if len(shape) == 3:
                num_frames = shape[0]
                h, w = shape[1], shape[2]
            elif len(shape) == 2:
                num_frames = 1
                h, w = shape[0], shape[1]
            else:
                return None

            return {
                "num_frames": num_frames,
                "shape": shape,
                "dtype": dtype,
                "h": h,
                "w": w,
            }

    except PermissionError:
        print(f"    Permission denied: {fpath}")
        return None
    except Exception as e:
        print(f"    Error reading {fpath}: {e}")
        return None


def validate_detector(file_info, detector_name):
    """Check that file dimensions match the expected detector config."""
    config = DETECTOR_CONFIG[detector_name]
    expected_h, expected_w = config["raw_shape"]
    return file_info["h"] == expected_h and file_info["w"] == expected_w


def build_manifest(data_dir, experiment_detector, experiment_filter=None):
    """Build the full manifest by scanning all experiment symlinks.

    Args:
        data_dir: Path to the data directory containing experiment symlinks.
        experiment_detector: Dict mapping symlink names to detector types.
        experiment_filter: Optional set of symlink names to process (None = all).
    """
    print("Building manifest...")
    print(f"Data directory: {data_dir}")

    # Find experiment symlinks
    symlinks = sorted([
        p for p in data_dir.iterdir()
        if p.is_symlink() and "--" in p.name
    ])

    print(f"Found {len(symlinks)} experiment symlinks.\n")

    experiments = []
    total_files = 0
    total_frames = 0
    skipped_files = 0

    for sl in symlinks:
        symlink_name = sl.name

        if experiment_filter and symlink_name not in experiment_filter:
            continue

        if symlink_name not in experiment_detector:
            print(f"  {symlink_name}: not in detector map, skipping.")
            continue

        detector = experiment_detector[symlink_name]
        experiment_id = symlink_name.split("--")[0]

        target = Path(os.readlink(sl))
        if not target.is_absolute():
            target = (sl.parent / target).resolve()

        print(f"  {symlink_name} ({detector})")

        hdf5_files = find_hdf5_files(target)
        print(f"    Found {len(hdf5_files)} HDF5 files")

        file_entries = []
        for fpath in hdf5_files:
            info = inspect_file(fpath)
            if info is None:
                skipped_files += 1
                continue

            if not validate_detector(info, detector):
                print(f"    Shape mismatch: {fpath.name} has "
                      f"{info['h']}x{info['w']}, expected "
                      f"{DETECTOR_CONFIG[detector]['raw_shape']}")
                skipped_files += 1
                continue

            # Store path relative to DATA_DIR for portability
            try:
                rel_path = str(fpath.relative_to(data_dir.resolve()))
            except ValueError:
                # If file is outside DATA_DIR, use path relative to symlink
                rel_path = f"{symlink_name}/{fpath.relative_to(target)}"

            file_entries.append({
                "path": f"data/{rel_path}",
                "num_frames": info["num_frames"],
                "shape": info["shape"],
                "dtype": info["dtype"],
            })
            total_frames += info["num_frames"]

        if file_entries:
            det_config = DETECTOR_CONFIG[detector]
            experiments.append({
                "experiment_id": experiment_id,
                "symlink": symlink_name,
                "detector": detector,
                "panel_shape": [det_config["panel_h"], det_config["panel_w"]],
                "num_panels": det_config["num_panels"],
                "image_key": IMAGE_KEY,
                "num_files": len(file_entries),
                "total_frames": sum(f["num_frames"] for f in file_entries),
                "files": file_entries,
            })
            total_files += len(file_entries)

        print(f"    Included {len(file_entries)} files, "
              f"{sum(f['num_frames'] for f in file_entries)} frames")

    manifest = {
        "version": "1.0",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "description": "LCLS X-ray diffraction/scattering image manifest for model training",
        "image_key": IMAGE_KEY,
        "detector_configs": DETECTOR_CONFIG,
        "summary": {
            "num_experiments": len(experiments),
            "num_files": total_files,
            "total_frames": total_frames,
            "skipped_files": skipped_files,
        },
        "experiments": experiments,
    }

    return manifest


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parent / "data"

    parser = argparse.ArgumentParser(
        description="Build a JSON manifest of HDF5 data files for model training."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=default_data_dir,
        help=f"Data directory containing experiment symlinks (default: {default_data_dir})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output manifest path (default: <data-dir>/manifest.json)",
    )
    parser.add_argument(
        "--detector-map", type=Path, default=None,
        help="YAML file mapping symlink names to detector types (default: built-in mapping)",
    )
    parser.add_argument(
        "--experiments", type=str, default=None,
        help="Comma-separated list of experiment symlink names to process (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir.resolve()
    output_path = args.output if args.output else data_dir / "manifest.json"

    if args.detector_map:
        experiment_detector = load_detector_map(args.detector_map)
    else:
        experiment_detector = _DEFAULT_EXPERIMENT_DETECTOR

    experiment_filter = None
    if args.experiments:
        experiment_filter = set(args.experiments.split(","))

    manifest = build_manifest(data_dir, experiment_detector, experiment_filter)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {output_path}")
    print(f"  Experiments: {manifest['summary']['num_experiments']}")
    print(f"  Files: {manifest['summary']['num_files']}")
    print(f"  Total frames: {manifest['summary']['total_frames']}")
    print(f"  Skipped files: {manifest['summary']['skipped_files']}")


if __name__ == "__main__":
    main()
