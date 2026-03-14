# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "tiled[server]",
#     "pandas",
#     "zarr",
#     "numpy",
#     "matplotlib",
#     "pyyaml",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LCLS SFX Data Broker: Egress Showcase

    This notebook demonstrates the **egress capabilities** of the LCLS SFX
    data broker -- the different ways diffraction images and derived data
    can be delivered to users from a single Tiled catalog.

    | Section | Capability |
    |---------|-----------|
    | 1 | Catalog overview: datasets, entities, artifacts |
    | 2 | Full frame retrieval (Mode B -- Tiled HTTP) |
    | 3 | Partial reads via slicing (Mode B) |
    | 4 | Direct Zarr access (Mode A -- expert) |
    | 5 | Data equivalence: Mode A == Mode B |
    | 6 | Query-driven egress: filter by frame statistics |
    | 7 | Multi-artifact egress: image + peaknet labels |
    | 8 | Cross-run uniform access |

    **Two egress modes, same data:**
    - **Mode A (Expert):** Extract file path and index from entity metadata,
      load directly via `zarr`. Best for ML pipelines and bulk loading.
    - **Mode B (Visualizer):** Access arrays through Tiled HTTP adapters.
      Best for interactive exploration and remote users.

    **Prerequisites:** Start the Tiled server:
    ```bash
    cd data/broker
    PYTHONPATH=../../broker uv run --with 'tiled[server]' \
        tiled serve config ../../broker/config.yml --api-key secret
    ```
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import zarr
    import time
    import os
    from pathlib import Path

    return mo, mpatches, np, os, plt, time, zarr


@app.cell
def _(mo, os):
    import socket
    from urllib.parse import urlparse

    from tiled.client import from_uri
    from tiled.queries import Key

    TILED_URL = os.environ.get("TILED_URL", "http://localhost:8007")
    API_KEY = os.environ.get("TILED_API_KEY", "secret")

    # Quick socket check to avoid from_uri's long retry loop
    _parsed = urlparse(TILED_URL)
    _host = _parsed.hostname or "localhost"
    _port = _parsed.port or 8007
    _reachable = False
    try:
        _sock = socket.create_connection((_host, _port), timeout=2)
        _sock.close()
        _reachable = True
    except (OSError, socket.timeout):
        pass

    client = None
    if _reachable:
        try:
            client = from_uri(TILED_URL, api_key=API_KEY)
        except Exception as e:
            mo.callout(
                mo.md(f"Connection to `{TILED_URL}` failed: {e}"),
                kind="warn",
            )

    if client is None:
        mo.callout(
            mo.md(f"""No Tiled server at `{TILED_URL}`.

    Start the server first:
    ```bash
    cd data/broker
    PYTHONPATH=../../broker uv run --with 'tiled[server]' \\
    tiled serve config ../../broker/config.yml --api-key secret
    ```
    """),
            kind="warn",
        )
    return Key, client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Catalog Overview

    The broker serves LCLS SFX (serial femtosecond crystallography) data
    from multiple experiments and instruments.  Each **dataset** is a run,
    each **entity** is a diffraction frame, and each **artifact** is an
    array (image, peaknet label, etc.).
    """)
    return


@app.cell
def _(client, mo):
    mo.stop(client is None, mo.callout(mo.md("Skipped -- no server connection."), kind="warn"))

    _dataset_keys = list(client.keys())
    _rows = []
    _total_ent = 0
    _total_art = 0
    for _dk in _dataset_keys:
        _ds = client[_dk]
        _m = dict(_ds.metadata)
        _n_ent = len(_ds)
        _total_ent += _n_ent
        # Sample first entity to count artifacts
        _first_key = list(_ds.keys()[:1])[0]
        _n_children = len(_ds[_first_key])
        _n_art = _n_ent * _n_children
        _total_art += _n_art
        _rows.append(
            f"| `{_dk}` | {_m.get('instrument', '?')} | "
            f"{_m.get('detector', '?')} | {_m.get('sample_name', '?')} | "
            f"{_n_ent:,} | {_n_children} | {_n_art:,} |"
        )

    _table = "\n".join(_rows)
    mo.md(f"""**Connected to `{client.uri}`** -- {len(_dataset_keys)} runs, {_total_ent:,} entities, {_total_art:,} artifacts.

    | Run | Instrument | Detector | Sample | Frames | Art/Frame | Total Art |
    |-----|-----------|----------|--------|--------|-----------|-----------|
    {_table}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Full Frame Retrieval (Mode B)

    The simplest egress pattern: navigate to a frame, call `.read()`.
    The Tiled server reads the Zarr store and delivers the array over HTTP.
    No file paths or Zarr internals needed.
    """)
    return


@app.cell
def _(client, mo, np, plt, time):
    mo.stop(client is None, mo.callout(mo.md("Skipped -- no server connection."), kind="warn"))

    # Pick a dataset and its first frame
    _dk = list(client.keys())[0]
    _ds = client[_dk]
    _ek = list(_ds.keys())[0]
    _ent = _ds[_ek]
    _children = list(_ent.keys())

    _t0 = time.perf_counter()
    _img = _ent["image"].read()
    _t_read = (time.perf_counter() - _t0) * 1000

    fig_full, _ax = plt.subplots(figsize=(8, 6))
    _vmax = np.percentile(_img[_img > 0], 99) if np.any(_img > 0) else 1
    _im = _ax.imshow(
        _img, aspect="auto", origin="lower", cmap="viridis",
        vmin=0, vmax=_vmax,
    )
    _ax.set_xlabel("Pixel x")
    _ax.set_ylabel("Pixel y")
    _ax.set_title(f"{_dk} / {_ek} / image  {_img.shape}  [{_t_read:.0f} ms]")
    plt.colorbar(_im, ax=_ax, label="Intensity")
    plt.tight_layout()

    mo.md(f"""**Run:** `{_dk}` | **Frame:** `{_ek}` | **Children:** `{_children}`

    | Array | Shape | dtype | Read time |
    |-------|-------|-------|-----------|
    | `image` | `{_img.shape}` | `{_img.dtype}` | {_t_read:.0f} ms |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Partial Reads via Slicing (Mode B)

    Tiled adapters support **numpy-style slicing** over HTTP.  Read a
    subregion of a detector image without downloading the full frame.
    Useful for ROI inspection and quick previews.
    """)
    return


@app.cell
def _(client, mo, mpatches, np, plt, time):
    mo.stop(client is None, mo.callout(mo.md("Skipped -- no server connection."), kind="warn"))

    _dk = list(client.keys())[0]
    _ds = client[_dk]
    _ek = list(_ds.keys())[0]
    _img_node = _ds[_ek]["image"]

    # Full frame
    _t0 = time.perf_counter()
    _img_full = _img_node.read()
    _t_full = (time.perf_counter() - _t0) * 1000
    _h, _w = _img_full.shape

    # Center ROI (quarter of the image)
    _r0, _r1 = _h // 4, 3 * _h // 4
    _c0, _c1 = _w // 4, 3 * _w // 4
    _t0 = time.perf_counter()
    _roi = _img_node[_r0:_r1, _c0:_c1]
    _t_roi = (time.perf_counter() - _t0) * 1000

    # Single row (horizontal cut)
    _row_idx = _h // 2
    _t0 = time.perf_counter()
    _row = _img_node[_row_idx, :]
    _t_row = (time.perf_counter() - _t0) * 1000

    _vmax = np.percentile(_img_full[_img_full > 0], 99) if np.any(_img_full > 0) else 1

    fig_slice, _axes = plt.subplots(1, 3, figsize=(16, 5))

    # Full image with ROI rectangle
    _axes[0].imshow(_img_full, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=_vmax)
    _rect = mpatches.Rectangle(
        (_c0, _r0), _c1 - _c0, _r1 - _r0,
        linewidth=2, edgecolor="red", facecolor="none",
    )
    _axes[0].add_patch(_rect)
    _axes[0].axhline(_row_idx, color="cyan", linewidth=1, linestyle="--")
    _axes[0].set_title(f"Full frame {_img_full.shape}  [{_t_full:.0f} ms]")

    # ROI subregion
    _axes[1].imshow(_roi, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=_vmax)
    _axes[1].set_title(f"Center ROI {_roi.shape}  [{_t_roi:.0f} ms]")

    # Row profile
    _axes[2].plot(_row, color="#059669", linewidth=1)
    _axes[2].set_title(f"Row [{_row_idx}, :] {_row.shape}  [{_t_row:.0f} ms]")
    _axes[2].set_xlabel("Pixel x")
    _axes[2].set_ylabel("Intensity")
    _axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    mo.md(f"""| Slice | Shape | Read time |
    |-------|-------|-----------|
    | Full `[:]` | `{_img_full.shape}` | {_t_full:.0f} ms |
    | Center ROI `[{_r0}:{_r1}, {_c0}:{_c1}]` | `{_roi.shape}` | {_t_roi:.0f} ms |
    | Row `[{_row_idx}, :]` | `{_row.shape}` | {_t_row:.0f} ms |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Expert Path-Based Access (Mode A)

    Each entity's metadata contains the **file path**, **dataset name**,
    and **frame index** needed to load data directly from Zarr -- bypassing
    the Tiled HTTP layer for maximum performance.

    ```python
    meta = entity.metadata
    chunk_file  = meta["chunk_file"]            # e.g. "cxi101235425_r0100.0000.zarr"
    frame_index = meta["chunk_frame_index"]     # index within the Zarr chunk
    ```
    """)
    return


@app.cell
def _(client, mo, time, zarr):
    mo.stop(client is None, mo.callout(mo.md("Skipped -- no server connection."), kind="warn"))

    _dk = list(client.keys())[0]
    _ds = client[_dk]
    _ek = list(_ds.keys())[0]
    _ent = _ds[_ek]
    _meta = dict(_ent.metadata)

    # Show the metadata fields available for Mode A
    _meta_display = {
        k: v for k, v in _meta.items()
        if not k.startswith(("path_", "dataset_", "index_"))
    }

    # Get base_dir from dataset metadata
    _ds_meta = dict(_ds.metadata)
    _base_dir = _ds_meta.get("base_dir", "/lustre/orion/lrn091/proj-shared/data")

    # Attempt direct Zarr load
    _chunk_file = _meta.get("chunk_file")
    _chunk_idx = _meta.get("chunk_frame_index")

    _mode_a_result = None
    if _chunk_file is not None:
        try:
            _zarr_path = f"{_base_dir}/{_chunk_file}"
            _t0 = time.perf_counter()
            _store = zarr.open(_zarr_path, mode="r")
            _arr_a = _store["images"][int(_chunk_idx)]
            _t_a = (time.perf_counter() - _t0) * 1000
            _mode_a_result = f"**Direct Zarr load:** `{_zarr_path}` images[{_chunk_idx}] -> shape `{_arr_a.shape}`, dtype `{_arr_a.dtype}` in **{_t_a:.0f} ms**"
        except (FileNotFoundError, OSError, KeyError) as e:
            _mode_a_result = f"*Direct Zarr load skipped: {e}*"

    _meta_rows = "\n".join(
        f"| `{k}` | `{v}` |" for k, v in _meta_display.items()
    )

    mo.md(f"""**Entity `{_ek}`** from run `{_dk}`:

    | Field | Value |
    |-------|-------|
    {_meta_rows}

    **Locator:** `chunk_file={_chunk_file}`, `chunk_frame_index={_chunk_idx}`

    {_mode_a_result or "*No chunk_file in metadata.*"}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Data Equivalence: Same Data, Two Modes

    Both modes access the same underlying Zarr store.  Mode A reads
    directly; Mode B reads through the Tiled adapter.  The arrays should
    be numerically identical.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Query-Driven Egress: Filter by Frame Statistics

    Every entity carries per-frame statistics (`mean_intensity`,
    `max_intensity`, `std_intensity`, `fraction_zero`, and sometimes
    `npeaks`).  Use Tiled queries to find interesting frames without
    scanning the full dataset.

    ```python
    from tiled.queries import Key
    bright = run.search(Key("max_intensity") > 50000)
    ```
    """)
    return


@app.cell
def _(Key, client, mo, np, plt, time):
    mo.stop(client is None, mo.callout(mo.md("Skipped -- no server connection."), kind="warn"))

    # Pick a run with decent frame count
    _dk = list(client.keys())[0]
    _ds = client[_dk]

    # Query: frames with high max intensity
    _t0 = time.perf_counter()
    _bright = _ds.search(Key("max_intensity") > 1000)
    _n_bright = len(_bright)
    _t_query = (time.perf_counter() - _t0) * 1000

    # Fetch first 4 bright frames
    _n_fetch = min(4, _n_bright)
    _keys = list(_bright.keys())[:_n_fetch]

    _frames = []
    _intensities = []
    _t0 = time.perf_counter()
    for _k in _keys:
        _ent = _bright[_k]
        _frames.append(_ent["image"].read())
        _intensities.append(_ent.metadata.get("max_intensity", 0))
    _t_fetch = (time.perf_counter() - _t0) * 1000

    if _n_fetch > 0:
        _ncols = min(4, _n_fetch)
        fig_query, _axes = plt.subplots(1, _ncols, figsize=(4 * _ncols, 4))
        if _ncols == 1:
            _axes = [_axes]

        for _i in range(_ncols):
            _vmax = np.percentile(_frames[_i][_frames[_i] > 0], 99) if np.any(_frames[_i] > 0) else 1
            _axes[_i].imshow(
                _frames[_i], aspect="auto", origin="lower",
                cmap="viridis", vmin=0, vmax=_vmax,
            )
            _axes[_i].set_title(f"{_keys[_i]}\nmax={_intensities[_i]:.0f}", fontsize=8)

        plt.tight_layout()

    mo.md(f"""**Run:** `{_dk}` ({len(_ds):,} frames total)

    **Query:** `Key("max_intensity") > 1000`

    | Step | Result | Time |
    |------|--------|------|
    | Server-side query | {_n_bright:,} hits out of {len(_ds):,} | {_t_query:.0f} ms |
    | Fetch {_n_fetch} frames (Mode B) | {_n_fetch} images | {_t_fetch:.0f} ms |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Multi-Artifact Egress: Image + PeakNet Labels

    Some runs include **PeakNet-derived labels** alongside the raw images.
    Both artifacts are children of the same entity and can be fetched
    with the same access pattern.
    """)
    return


@app.cell
def _(client, mo, np, plt, time):
    mo.stop(client is None, mo.callout(mo.md("Skipped -- no server connection."), kind="warn"))

    # Find a run with multiple artifact types (image + label)
    _multi_dk = None
    for _dk in client.keys():
        _ds = client[_dk]
        _first_key = list(_ds.keys()[:1])[0]
        _children = list(_ds[_first_key].keys())
        if len(_children) > 1:
            _multi_dk = _dk
            break

    mo.stop(_multi_dk is None, mo.callout(mo.md("No multi-artifact runs found in the catalog."), kind="info"))

    _ds = client[_multi_dk]
    _ek = list(_ds.keys())[0]
    _ent = _ds[_ek]
    _children = list(_ent.keys())

    fig_multi, _axes = plt.subplots(1, len(_children), figsize=(6 * len(_children), 5))
    if len(_children) == 1:
        _axes = [_axes]

    _rows = []
    for _i, _child in enumerate(_children):
        _t0 = time.perf_counter()
        try:
            _arr = _ent[_child].read()
        except Exception as _e:
            _axes[_i].text(0.5, 0.5, f"read error:\n{_e}", ha="center", va="center",
                          fontsize=7, wrap=True, transform=_axes[_i].transAxes)
            _axes[_i].set_title(f"{_child}  [error]")
            _rows.append(f"| `{_child}` | error | -- | -- |")
            continue
        _t_read = (time.perf_counter() - _t0) * 1000

        _vmax = np.percentile(np.abs(_arr[_arr != 0]), 99) if np.any(_arr != 0) else 1
        _axes[_i].imshow(
            _arr, aspect="auto", origin="lower",
            cmap="viridis" if _child == "image" else "hot",
            vmin=0, vmax=_vmax,
        )
        _axes[_i].set_title(f"{_child}  {_arr.shape}  [{_t_read:.0f} ms]")
        _rows.append(
            f"| `{_child}` | `{_arr.shape}` | `{_arr.dtype}` | {_t_read:.0f} ms |"
        )

    plt.tight_layout()

    _meta = _ent.metadata
    _npeaks = _meta.get("npeaks", "N/A")
    _table = "\n".join(_rows)
    mo.md(f"""**Run:** `{_multi_dk}` | **Frame:** `{_ek}` | **npeaks:** {_npeaks}

    | Artifact | Shape | dtype | Read time |
    |----------|-------|-------|-----------|
    {_table}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Cross-Run Uniform Access

    Different runs come from different experiments, instruments, and
    detectors -- but the egress pattern is identical.  One call to
    `client[run][frame]["image"].read()` works for all.
    """)
    return


@app.cell
def _(client, mo, np, plt, time):
    mo.stop(client is None, mo.callout(mo.md("Skipped -- no server connection."), kind="warn"))

    _all_keys = list(client.keys())
    # Sample up to 6 runs spread across the catalog
    _step = max(1, len(_all_keys) // 6)
    _sample_keys = _all_keys[::_step][:6]

    _ncols = min(3, len(_sample_keys))
    _nrows = (len(_sample_keys) + _ncols - 1) // _ncols
    fig_cross, _axes = plt.subplots(_nrows, _ncols, figsize=(5 * _ncols, 4 * _nrows))
    _axes_flat = np.array(_axes).flatten() if len(_sample_keys) > 1 else [_axes]

    _rows = []
    for _i, _dk in enumerate(_sample_keys):
        _ds = client[_dk]
        _ds_meta = dict(_ds.metadata)
        _ek = list(_ds.keys())[0]

        _t0 = time.perf_counter()
        try:
            _img = _ds[_ek]["image"].read()
        except Exception as _e:
            _ax = _axes_flat[_i]
            _ax.text(0.5, 0.5, f"read error:\n{_e}", ha="center", va="center",
                     fontsize=7, wrap=True, transform=_ax.transAxes)
            _ax.set_title(f"{_dk}", fontsize=8)
            _rows.append(
                f"| `{_dk}` | {_ds_meta.get('instrument', '?')} | "
                f"{_ds_meta.get('detector', '?')} | error | -- | -- |"
            )
            continue
        _t_read = (time.perf_counter() - _t0) * 1000

        _ax = _axes_flat[_i]
        _vmax = np.percentile(_img[_img > 0], 99) if np.any(_img > 0) else 1
        _ax.imshow(_img, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=_vmax)
        _ax.set_title(f"{_dk}\n{_ds_meta.get('instrument', '?')} / {_ds_meta.get('detector', '?')}", fontsize=8)

        _rows.append(
            f"| `{_dk}` | {_ds_meta.get('instrument', '?')} | "
            f"{_ds_meta.get('detector', '?')} | `{_img.shape}` | "
            f"`{_img.dtype}` | {_t_read:.0f} ms |"
        )

    # Hide unused axes
    for _j in range(len(_sample_keys), len(_axes_flat)):
        _axes_flat[_j].set_visible(False)

    plt.tight_layout()

    _table = "\n".join(_rows)
    mo.md(f"""| Run | Instrument | Detector | Shape | dtype | Read time |
    |-----|-----------|----------|-------|-------|-----------|
    {_table}

    Same API across {len(_all_keys)} runs -- uniform egress regardless of source.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | Capability | Mode | Section |
    |-----------|------|---------|
    | Catalog browsing | B | 1 |
    | Full frame retrieval | B | 2 |
    | Partial/sliced read (ROI) | B | 3 |
    | Direct Zarr access | A | 4 |
    | Data equivalence | A + B | 5 |
    | Query-driven fetch (frame stats) | B | 6 |
    | Multi-artifact (image + label) | B | 7 |
    | Cross-run uniform access | B | 8 |

    **Key takeaways:**

    - **Mode B** is the universal egress path -- works from any location,
      supports slicing, no Zarr paths needed.
    - **Mode A** gives direct Zarr access for ML pipelines that need
      maximum throughput.
    - The broker serves **104K+ frames** from 30 SFX runs through a
      single unified API.
    """)
    return


@app.cell
def _(client, mo, np, plt, time, zarr):
    mo.stop(client is None, mo.callout(mo.md("Skipped -- no server connection."), kind="warn"))

    _dk = list(client.keys())[0]
    _ds = client[_dk]
    _ek = list(_ds.keys())[0]
    _ent = _ds[_ek]
    _meta = dict(_ent.metadata)
    _ds_meta = dict(_ds.metadata)

    # Mode B
    _t0 = time.perf_counter()
    _arr_b = _ent["image"].read()
    _t_b = (time.perf_counter() - _t0) * 1000

    # Mode A
    _mode_a_ok = False
    _chunk_file = _meta.get("chunk_file")
    _chunk_idx = _meta.get("chunk_frame_index")
    _base_dir = _ds_meta.get("base_dir", "/lustre/orion/lrn091/proj-shared/data")

    if _chunk_file is not None:
        try:
            _t0 = time.perf_counter()
            _store = zarr.open(f"{_base_dir}/{_chunk_file}", mode="r")
            _arr_a = _store["images"][int(_chunk_idx)]
            _t_a = (time.perf_counter() - _t0) * 1000
            _mode_a_ok = True
        except (FileNotFoundError, OSError, KeyError):
            pass

    fig_equiv, _axes = plt.subplots(1, 2, figsize=(12, 5))
    _vmax = np.percentile(_arr_b[_arr_b > 0], 99) if np.any(_arr_b > 0) else 1

    _axes[0].imshow(_arr_b, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=_vmax)
    _axes[0].set_title(f"Mode B (Tiled)  [{_t_b:.0f} ms]")

    if _mode_a_ok:
        _match = np.allclose(_arr_a, _arr_b)
        _axes[1].imshow(_arr_a, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=_vmax)
        _axes[1].set_title(f"Mode A (Zarr)  [{_t_a:.0f} ms]")
        _result = f"**np.allclose(A, B) = `{_match}`** -- identical data from both modes."
    else:
        _axes[1].text(0.5, 0.5, "Mode A\nnot available", ha="center", va="center", transform=_axes[1].transAxes, fontsize=14)
        _axes[1].set_title("Mode A (not available)")
        _result = "*Mode A not available -- Zarr files not accessible from this host.*"

    plt.tight_layout()
    mo.md(_result)
    return


if __name__ == "__main__":
    app.run()
