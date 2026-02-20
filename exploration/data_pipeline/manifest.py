"""Manifest reader for the LCLS diffraction data manifest.

Provides utilities to load the JSON manifest and filter files by experiment
or detector type.
"""

import json
from pathlib import Path


def load_manifest(path):
    """Load and validate a manifest JSON file.

    Args:
        path: Path to the manifest.json file.

    Returns:
        dict: The parsed manifest.

    Raises:
        FileNotFoundError: If the manifest file doesn't exist.
        ValueError: If the manifest is missing required fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path) as f:
        manifest = json.load(f)

    # Basic validation
    required = ["version", "experiments", "detector_configs"]
    for key in required:
        if key not in manifest:
            raise ValueError(f"Manifest missing required field: '{key}'")

    if not manifest["experiments"]:
        raise ValueError("Manifest has no experiments")

    return manifest


def list_files(manifest, experiments=None, detectors=None):
    """List data files from the manifest, optionally filtered.

    Args:
        manifest: Parsed manifest dict (from load_manifest).
        experiments: Optional list of experiment IDs to include.
            If None, includes all experiments.
        detectors: Optional list of detector types to include
            (e.g., ["jungfrau_4m", "epix10k_2m"]).
            If None, includes all detectors.

    Returns:
        list of dict: Each dict has keys:
            - path (str): Relative path to the HDF5 file
            - num_frames (int): Number of frames in the file
            - shape (list): Dataset shape [N, H, W]
            - dtype (str): Data type
            - experiment_id (str): Parent experiment ID
            - detector (str): Detector type name
            - panel_shape (list): [panel_h, panel_w]
            - num_panels (int): Number of panels per frame
            - image_key (str): HDF5 key for image data
    """
    results = []

    for exp in manifest["experiments"]:
        # Filter by experiment ID
        if experiments is not None and exp["experiment_id"] not in experiments:
            continue

        # Filter by detector type
        if detectors is not None and exp["detector"] not in detectors:
            continue

        for file_entry in exp["files"]:
            results.append({
                **file_entry,
                "experiment_id": exp["experiment_id"],
                "detector": exp["detector"],
                "panel_shape": exp["panel_shape"],
                "num_panels": exp["num_panels"],
                "image_key": exp["image_key"],
            })

    return results
