"""Visualize sample diffraction images from each experiment.

For each experiment, picks ~3 files across different runs, extracts 2-3 frames,
and saves visualizations with multiple clipping strategies.

Usage:
    uv run --with h5py --with matplotlib --with numpy exploration/scripts/visualize_samples.py
"""

import os
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TMP_DIR = Path("/tmp/lcls_sample_images")
SAMPLE_DIR = DATA_DIR / "sample-images"

# Map of symlink name -> (image HDF5 key, description)
# Only include symlinks with actual 2D diffraction images.
EXPERIMENT_CONFIG = {
    "cxi101235425--cheetah-hdf5": {
        "image_key": "/entry_1/data_1/data",
        "desc": "Cheetah SFX (Jungfrau 4096x1024)",
    },
    "cxil1005322--cheetah-hdf5": {
        "image_key": "/entry_1/data_1/data",
        "desc": "Cheetah nucleic acid scattering (Jungfrau 4096x1024)",
    },
    "cxil1015922--cheetah-hdf5": {
        "image_key": "/entry_1/data_1/data",
        "desc": "Cheetah ligand dissociation (Jungfrau 4096x1024)",
    },
    "cxilw5019--psocake": {
        "image_key": "/entry_1/data_1/data",
        "desc": "psocake SFX GPCRs (4096x1024)",
    },
    "mfx100903824--results-cxi": {
        "image_key": "/entry_1/data_1/data",
        "desc": "OM cryptochrome (ePix10k2M 5632x384)",
    },
    "mfx101211025--cheetah-hdf5": {
        "image_key": "/entry_1/data_1/data",
        "desc": "Cheetah photolyase (16384x1024)",
    },
    "mfxp22421--cheetah-hdf5": {
        "image_key": "/entry_1/data_1/data",
        "desc": "Cheetah fixed-target SFX (ePix10k2M 5632x384)",
    },
    "mfxx49820--lute-drcomp-f32": {
        "image_key": "/entry_1/data_1/data",
        "desc": "LUTE DrComp automated droplet (ePix10k2M 5632x384)",
    },
    "prjcwang31--userdata-cheetah": {
        "image_key": "/entry_1/data_1/data",
        "desc": "Cheetah user data (ePix10k2M 5632x384)",
    },
    "prjcwang31--userdata-psocake": {
        "image_key": "/entry_1/data_1/data",
        "desc": "psocake user data (1920x1920 square)",
    },
}

MAX_FILES_PER_EXPERIMENT = 3
MAX_FRAMES_PER_FILE = 3


def find_data_files(symlink_path, max_files=MAX_FILES_PER_EXPERIMENT):
    """Find .cxi/.h5 files, trying to sample from different run directories."""
    extensions = {".cxi", ".h5", ".hdf5"}
    by_dir = {}

    for dirpath, dirnames, filenames in os.walk(symlink_path):
        depth = dirpath.replace(str(symlink_path), "").count(os.sep)
        if depth > 4:
            dirnames.clear()
            continue

        for fname in filenames:
            if Path(fname).suffix.lower() in extensions:
                fpath = Path(dirpath) / fname
                # Skip very small files (likely masks or metadata)
                try:
                    if fpath.stat().st_size < 5_000_000:  # < 5 MB
                        continue
                except OSError:
                    continue
                by_dir.setdefault(dirpath, []).append(fpath)

    # Prefer .cxi files over .h5 (summary/mask files are often .h5)
    def sort_key(f):
        return (0 if f.suffix.lower() == ".cxi" else 1, f.name)

    # Pick one file from each directory to maximize variety
    selected = []
    for dirpath in sorted(by_dir.keys()):
        files = sorted(by_dir[dirpath], key=sort_key)
        selected.append(files[0])
        if len(selected) >= max_files:
            break

    # If we still need more, add from the first directory
    if len(selected) < max_files:
        for dirpath in sorted(by_dir.keys()):
            for f in sorted(by_dir[dirpath], key=sort_key):
                if f not in selected:
                    selected.append(f)
                    if len(selected) >= max_files:
                        break
            if len(selected) >= max_files:
                break

    return selected


def pick_frame_indices(n_frames):
    """Pick up to MAX_FRAMES_PER_FILE evenly spaced frame indices."""
    if n_frames <= 0:
        return []
    if n_frames <= MAX_FRAMES_PER_FILE:
        return list(range(n_frames))
    # first, middle, last
    return [0, n_frames // 2, n_frames - 1]


def save_image(img, title, filepath, method="mean4std"):
    """Save a single image with the given clipping method."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=100)

    if method == "mean4std":
        mu = np.nanmean(img)
        sigma = np.nanstd(img)
        vmin, vmax = mu, mu + 4 * sigma
        label = f"mean={mu:.1f}, mean+4std={vmax:.1f}"
    elif method == "log":
        # Shift so minimum is 1, then log
        img_shifted = img - np.nanmin(img) + 1
        img = np.log10(img_shifted)
        vmin, vmax = np.nanpercentile(img, 1), np.nanpercentile(img, 99.5)
        label = f"log10 scale, clip [{vmin:.1f}, {vmax:.1f}]"
    elif method == "pct99":
        vmin = np.nanpercentile(img, 1)
        vmax = np.nanpercentile(img, 99)
        label = f"1st-99th percentile [{vmin:.1f}, {vmax:.1f}]"
    else:
        vmin, vmax = np.nanmin(img), np.nanmax(img)
        label = f"full range [{vmin:.1f}, {vmax:.1f}]"

    im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
    ax.set_title(f"{title}\n{label}", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def process_experiment(symlink_name, config):
    """Process one experiment: find files, extract frames, save images."""
    symlink_path = DATA_DIR / symlink_name
    if not symlink_path.exists():
        print(f"  Symlink not found, skipping.")
        return 0

    target = symlink_path.resolve()
    image_key = config["image_key"]
    desc = config["desc"]

    print(f"\n{'='*60}")
    print(f"  {symlink_name}")
    print(f"  {desc}")
    print(f"  Image key: {image_key}")
    print(f"{'='*60}")

    files = find_data_files(target)
    if not files:
        print("  No suitable data files found.")
        return 0

    count = 0
    for fpath in files:
        rel = fpath.relative_to(target)
        # Extract a short run label from the path
        run_label = str(rel).replace("/", "_").replace(".cxi", "").replace(".h5", "")

        try:
            with h5py.File(fpath, "r") as f:
                if image_key not in f:
                    print(f"  {rel}: key '{image_key}' not found, skipping.")
                    continue

                ds = f[image_key]
                shape = ds.shape
                print(f"  {rel}: shape={shape}, dtype={ds.dtype}")

                if len(shape) == 2:
                    # Single image
                    indices = [None]  # None means read the whole thing
                elif len(shape) >= 3:
                    n_frames = shape[0]
                    indices = pick_frame_indices(n_frames)
                else:
                    print(f"    Unexpected shape, skipping.")
                    continue

                for idx in indices:
                    if idx is None:
                        frame = ds[:]
                        frame_label = "full"
                    else:
                        frame = ds[idx]
                        frame_label = f"frame{idx:04d}"

                    # Convert to float for stats
                    frame = np.array(frame, dtype=np.float32)

                    # If still 3D+ after slicing (multi-panel), vstack panels
                    while frame.ndim > 2:
                        frame = frame.reshape(-1, frame.shape[-1])

                    exp_id = symlink_name.split("--")[0]
                    title = f"{exp_id} | {run_label} | {frame_label}"
                    base_name = f"{exp_id}__{run_label}__{frame_label}"

                    # Save with all three methods
                    for method in ["mean4std", "log", "pct99"]:
                        fname = f"{base_name}__{method}.png"

                        # Save to /tmp
                        tmp_path = TMP_DIR / exp_id / fname
                        save_image(frame.copy(), title, tmp_path, method)

                        # Save to data/sample-images (only mean4std)
                        if method == "mean4std":
                            sample_path = SAMPLE_DIR / exp_id / fname
                            save_image(frame.copy(), title, sample_path, method)

                    count += 1
                    print(f"    Saved {frame_label}: shape={frame.shape}, "
                          f"min={np.nanmin(frame):.1f}, max={np.nanmax(frame):.1f}, "
                          f"mean={np.nanmean(frame):.1f}, std={np.nanstd(frame):.1f}")

        except PermissionError:
            print(f"  {rel}: Permission denied.")
        except Exception as e:
            print(f"  {rel}: Error - {e}")

    return count


def main():
    print("Sample Image Visualization")
    print(f"Output: {TMP_DIR} (all), {SAMPLE_DIR} (curated)")

    total = 0
    for symlink_name, config in EXPERIMENT_CONFIG.items():
        n = process_experiment(symlink_name, config)
        total += n

    print(f"\n\nDone. Saved {total} sample images total.")
    print(f"  All images:    {TMP_DIR}/")
    print(f"  Sample images: {SAMPLE_DIR}/")


if __name__ == "__main__":
    main()
