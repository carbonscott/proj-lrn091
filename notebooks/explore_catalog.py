# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "tiled[client]",
#     "pandas",
#     "numpy",
#     "zarr",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def cell_title():
    import marimo as mo

    mo.md(
        """
        # SFX Catalog Explorer

        This notebook explores the SFX crystallography catalog served by
        [Tiled](https://blueskyproject.io/tiled/).  The catalog contains:

        - **44 runs** (datasets) from CXI and MFX instruments
        - **~440K frames** of assembled detector images
        - **~458K artifacts** (images + peak labels)

        **Hierarchy**: dataset = run, entity = frame, artifact = image or label.

        ## How to start the Tiled server

        ```bash
        cd data/broker
        uv run --with 'tiled[server]' tiled serve config config.yml --api-key secret
        ```

        Then run this notebook:

        ```bash
        uv run --with marimo --with 'tiled[client]' --with pandas --with numpy \\
            --with zarr --with matplotlib marimo run scripts/explore_catalog.py
        ```
        """
    )
    return (mo,)


@app.cell
def cell_imports():
    import os
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import zarr
    from matplotlib.colors import LogNorm
    from tiled.queries import Key

    return Key, LogNorm, np, os, pd, plt, time, zarr


@app.cell
def cell_connect(mo):
    from tiled.client import from_uri

    client = from_uri("http://localhost:8007", api_key="secret")

    _total_datasets = len(client)
    _total_frames = sum(len(client[k]) for k in client)

    mo.md(
        f"""
        ## Connection

        Connected to Tiled at `http://localhost:8007`

        | Metric | Value |
        |--------|-------|
        | Datasets (runs) | **{_total_datasets}** |
        | Total frames | **{_total_frames:,}** |
        """
    )
    return (client,)


@app.cell
def cell_overview(client, mo, pd):
    _rows = []
    for _key in client:
        _ds = client[_key]
        _meta = dict(_ds.metadata)
        _rows.append(
            {
                "Run": _key,
                "Instrument": _meta.get("instrument", ""),
                "Detector": _meta.get("detector", ""),
                "Data Type": _meta.get("data_type", ""),
                "Frames": _meta.get("num_frames", len(_ds)),
                "Sample": _meta.get("sample_name", ""),
            }
        )

    df_overview = pd.DataFrame(_rows)
    mo.vstack([mo.md("## Catalog Overview"), mo.ui.table(df_overview)])
    return (df_overview,)


@app.cell
def cell_run_queries(Key, client, mo):
    _queries = {
        'Jungfrau 4M runs':        Key("detector") == "jungfrau_4m",
        'PeakNet labeled runs':    Key("data_type") == "peaknet_labeled",
        'Cherezov experiments':    Key("pi") == "Cherezov",
        'Large runs (>10K frames)': Key("num_frames") > 10000,
    }

    _lines = []
    for _label, _query in _queries.items():
        _results = client.search(_query)
        _keys = list(_results)
        _lines.append(f"| {_label} | {len(_results)} | `{_keys[:5]}`{'...' if len(_keys) > 5 else ''} |")

    mo.md(
        "## Run-Level Queries\n\n"
        "| Query | Matches | Keys (sample) |\n"
        "|-------|---------|----------------|\n"
        + "\n".join(_lines)
    )
    return


@app.cell
def cell_frame_queries(Key, client, mo):
    # Pick an assembled run
    _asm_key = "cxilw5019_r0017"
    _run = client[_asm_key]
    _total = len(_run)

    _bright = _run.search(Key("max_intensity") > 5000)
    _sparse = _run.search(Key("fraction_zero") > 0.8)

    _lines = [
        f"**Run `{_asm_key}`** — {_total:,} frames\n",
        f"| Query | Matches | % |",
        f"|-------|---------|---|",
        f"| `max_intensity > 5000` | {len(_bright):,} | {100*len(_bright)/_total:.1f}% |",
        f"| `fraction_zero > 0.8`  | {len(_sparse):,} | {100*len(_sparse)/_total:.1f}% |",
    ]

    # Pick a peaknet run
    _pn_key = "mfxl1025422_r0309"
    _prun = client[_pn_key]
    _ptotal = len(_prun)
    _peak_rich = _prun.search(Key("npeaks") > 30)

    _lines += [
        f"\n**Run `{_pn_key}`** (peaknet) — {_ptotal:,} frames\n",
        f"| Query | Matches | % |",
        f"|-------|---------|---|",
        f"| `npeaks > 30` | {len(_peak_rich):,} | {100*len(_peak_rich)/_ptotal:.1f}% |",
    ]

    mo.md("## Frame-Level Queries\n\n" + "\n".join(_lines))
    return


@app.cell
def cell_entity(client, mo):
    run_key = "cxilw5019_r0017"
    _run = client[run_key]

    _entity_key = list(_run)[:1][0]
    entity = _run[_entity_key]
    _meta = dict(entity.metadata)

    _meta_lines = "\n".join(f"  {k}: {v}" for k, v in _meta.items())

    mo.md(
    f"""
    ## Entity Browsing

    **Run**: `{run_key}` → **Entity**: `{_entity_key}`

    Children (artifacts): `{list(entity)}`

    ### Metadata
        ```
    {_meta_lines}
        ```

    The **locator fields** (`path_image`, `dataset_image`, `index_image`)
    tell you exactly where the raw data lives on disk.

    - **Mode A** — use the locator to open the Zarr file directly (fast,
      requires filesystem access).
    - **Mode B** — read through the Tiled HTTP adapter (`entity["image"][:]`).
        """
    )
    return entity, run_key


@app.cell
def cell_mode_a(client, entity, mo, np, os, run_key, time, zarr):
    # Base directories matching readable_storage in config.yml
    _BASE_DIRS = {
        "assembled":       "/sdf/data/lcls/ds/prj/prjdat21/results/cwang31/proj-lrn091/data/assembled",
        "peaknet_labeled": "/sdf/data/lcls/ds/prj/prjdat21/results/cwang31/proj-lrn091/data/peaknet10k",
    }

    _run = client[run_key]
    _data_type = _run.metadata.get("data_type", "assembled")
    _base_dir = _BASE_DIRS[_data_type]

    _meta = dict(entity.metadata)
    _zarr_path = os.path.join(_base_dir, _meta["path_image"])
    _dataset   = _meta["dataset_image"]
    _index     = int(_meta["index_image"])

    _t0 = time.perf_counter()
    _store = zarr.open(_zarr_path, mode="r")
    frame_a = np.array(_store[_dataset][_index])
    _dt_a = time.perf_counter() - _t0

    mo.md(
        f"""
        ## Mode A — Direct Zarr Loading

        | Field | Value |
        |-------|-------|
        | Zarr path | `{_zarr_path}` |
        | Dataset   | `{_dataset}` |
        | Index     | `{_index}` |
        | Shape     | `{frame_a.shape}` |
        | dtype     | `{frame_a.dtype}` |
        | Time      | `{_dt_a:.3f} s` |
        """
    )
    return (frame_a,)


@app.cell
def cell_mode_b(entity, frame_a, mo, np, time):
    try:
        _t0 = time.perf_counter()
        _frame_b = entity["image"][:]
        _dt_b = time.perf_counter() - _t0

        _match = np.allclose(frame_a, _frame_b)

        _output = (
            f"## Mode B — Tiled HTTP Loading\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| Shape     | `{_frame_b.shape}` |\n"
            f"| dtype     | `{_frame_b.dtype}` |\n"
            f"| Time      | `{_dt_b:.3f} s` |\n"
            f"| **allclose(A, B)** | **{_match}** |"
        )
    except Exception as _e:
        _output = (
            f"## Mode B — Tiled HTTP Loading\n\n"
            f"**Error**: `{_e}`\n\n"
            f"This typically means the Tiled server adapter cannot serve this "
            f"array.  Mode A (direct Zarr) still works."
        )

    mo.md(_output)
    return


@app.cell
def cell_viz_frame(LogNorm, entity, frame_a, mo, plt, run_key):
    _meta = dict(entity.metadata)
    _pos = frame_a[frame_a > 0]

    _fig, _ax = plt.subplots(figsize=(8, 8))
    if _pos.size > 0:
        _im = _ax.imshow(
            frame_a,
            cmap="viridis",
            norm=LogNorm(vmin=max(float(_pos.min()), 1), vmax=float(frame_a.max())),
            # vmin=frame_a.mean(), vmax=frame_a.mean() + frame_a.std()*3,
        )
        _fig.colorbar(_im, ax=_ax, label="Intensity (log)")
    else:
        _ax.imshow(frame_a, cmap="viridis")
    _ax.set_title(f"{run_key} / frame {_meta.get('frame_index', '?')}")
    plt.tight_layout()

    mo.vstack([mo.md("## Visualization — Single Frame"), _fig])
    return


@app.cell
def cell_viz_distributions(client, mo, plt):
    # Assembled run
    _asm_key = "cxilw5019_r0017"
    _asm_run = client[_asm_key]
    _asm_max, _asm_fzero = [], []
    for _ek in list(_asm_run)[:2000]:
        _m = dict(_asm_run[_ek].metadata)
        _asm_max.append(float(_m["max_intensity"]))
        _asm_fzero.append(float(_m["fraction_zero"]))

    # PeakNet run
    _pn_key = "mfxl1025422_r0309"
    _pn_run = client[_pn_key]
    _pn_npeaks = []
    for _ek in list(_pn_run)[:2000]:
        _m = dict(_pn_run[_ek].metadata)
        _pn_npeaks.append(int(_m["npeaks"]))

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))

    _axes[0].hist(_asm_max, bins=50, edgecolor="black", linewidth=0.3)
    _axes[0].set_xlabel("max_intensity")
    _axes[0].set_ylabel("Count")
    _axes[0].set_title(f"{_asm_key} — max_intensity")

    _axes[1].hist(_asm_fzero, bins=50, edgecolor="black", linewidth=0.3)
    _axes[1].set_xlabel("fraction_zero")
    _axes[1].set_ylabel("Count")
    _axes[1].set_title(f"{_asm_key} — fraction_zero")

    _axes[2].hist(_pn_npeaks, bins=50, edgecolor="black", linewidth=0.3)
    _axes[2].set_xlabel("npeaks")
    _axes[2].set_ylabel("Count")
    _axes[2].set_title(f"{_pn_key} — npeaks")

    plt.tight_layout()

    mo.vstack([
        mo.md(
            "## Intensity Distributions\n\n"
            "Sampled up to 2000 frames per run (metadata only — no image loading)."
        ),
        _fig,
    ])
    return


@app.cell
def cell_viz_detector(client, df_overview, mo, pd, plt):
    _detectors = df_overview["Detector"].unique()
    _data = {_det: [] for _det in _detectors}

    for _det in _detectors:
        _det_runs = [r for r, d in zip(df_overview["Run"], df_overview["Detector"]) if d == _det]
        _sampled = 0
        for _rk in _det_runs:
            _run = client[_rk]
            for _ek in list(_run)[:200]:
                _m = dict(_run[_ek].metadata)
                _data[_det].append(
                    {
                        "detector": _det,
                        "mean_intensity": float(_m.get("mean_intensity", 0)),
                        "std_intensity":  float(_m.get("std_intensity", 0)),
                    }
                )
                _sampled += 1
            if _sampled >= 500:
                break

    _records = [r for v in _data.values() for r in v]

    if _records:
        _df = pd.DataFrame(_records)

        _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

        _det_labels = sorted(_df["detector"].unique())

        _mean_groups = [_df[_df["detector"] == d]["mean_intensity"].values for d in _det_labels]
        _axes[0].boxplot(_mean_groups, labels=_det_labels, vert=True)
        _axes[0].set_ylabel("mean_intensity")
        _axes[0].set_title("Mean Intensity by Detector")
        _axes[0].tick_params(axis="x", rotation=20)

        _std_groups = [_df[_df["detector"] == d]["std_intensity"].values for d in _det_labels]
        _axes[1].boxplot(_std_groups, labels=_det_labels, vert=True)
        _axes[1].set_ylabel("std_intensity")
        _axes[1].set_title("Std Intensity by Detector")
        _axes[1].tick_params(axis="x", rotation=20)

        plt.tight_layout()

        _msg = "Sampled ~500 frames per detector type across runs."
    else:
        _fig = None
        _msg = "No data available."

    _elements = [mo.md(f"## Detector Comparison\n\n{_msg}")]
    if _fig is not None:
        _elements.append(_fig)
    mo.vstack(_elements)
    return


@app.cell
def cell_summary(mo):
    mo.md("""
    ## Summary

    This catalog enables **metadata-driven frame selection** for ML training:

    - **Run-level filtering** — select by instrument, detector, PI, data type,
      or run size.
    - **Frame-level filtering** — find bright frames, sparse frames, or
      peak-rich frames without loading images.
    - **Reproducible datasets** — queries return deterministic subsets
      identified by UID.
    - **Two access modes** — Mode A (direct Zarr) for performance on local
      filesystems; Mode B (Tiled HTTP) for remote or notebook access.
    - **Per-frame statistics** — `mean_intensity`, `max_intensity`,
      `std_intensity`, `fraction_zero`, `npeaks` available as metadata.

    Next steps: use these queries to construct balanced training sets
    for PeakNet, filtering by intensity and peak count.
    """)
    return


if __name__ == "__main__":
    app.run()
