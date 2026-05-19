"""
CaImAn-backed pipeline for LBM data.

Produces the same on-disk layout as ``lbm_suite2p_python.pipeline()`` so the
results can be consumed by mbo studio / lbm_suite2p_python tooling. CaImAn
provides motion correction and CNMF source extraction; outputs are mapped
into suite2p-style files (``data.bin``, ``ops.npy``, ``stat.npy``,
``iscell.npy``, ``F.npy``, ``Fneu.npy``, ``spks.npy``, ``dff.npy``,
``roi_stats.npy``) and the lsp post-processing helpers (figures, volume
stats) run on top of them.
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np

from lbm_caiman_python.default_ops import default_ops

logger = logging.getLogger(__name__)

PIPELINE_TAGS = ("plane", "roi", "z", "plane_", "roi_", "z_")

LAZY_TYPES = (
    "TiffArray", "ScanImageArray", "LBMArray", "PiezoArray",
    "Suite2pArray", "H5Array", "ZarrArray", "NumpyArray",
    "SinglePlaneArray", "ImageJHyperstackArray", "BinArray",
)


def _get_version() -> str:
    try:
        from lbm_caiman_python import __version__
        return __version__
    except Exception:
        return "0.0.0"


def _get_caiman_version() -> str:
    try:
        import caiman
        return getattr(caiman, "__version__", "unknown")
    except ImportError:
        return "not installed"


def _is_lazy_array(obj) -> bool:
    return type(obj).__name__ in LAZY_TYPES


def _resolve_input_path(path):
    """Resolve a file or directory path for imread."""
    path = Path(path)
    if path.is_dir():
        from mbo_utilities import get_files
        files = get_files(str(path), str_contains="tif", max_depth=1)
        if not files:
            raise FileNotFoundError(f"no tiff files found in {path}")
        return files
    return path


def _get_num_planes(arr) -> int:
    if hasattr(arr, "num_planes"):
        return arr.num_planes
    if hasattr(arr, "shape5d"):
        return int(arr.shape5d[2])
    if hasattr(arr, "shape") and arr.ndim == 4:
        return arr.shape[1]
    return 1


def add_processing_step(ops, step_name, input_files=None, duration_seconds=None, extra=None):
    """Append a processing step record to ``ops['processing_history']``."""
    if "processing_history" not in ops:
        ops["processing_history"] = []

    step_record = {
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        "lbm_caiman_python_version": _get_version(),
        "caiman_version": _get_caiman_version(),
    }
    if input_files is not None:
        step_record["input_files"] = (
            [input_files] if isinstance(input_files, str)
            else [str(f) for f in input_files]
        )
    if duration_seconds is not None:
        step_record["duration_seconds"] = round(duration_seconds, 2)
    if extra is not None:
        step_record["extra"] = extra

    ops["processing_history"].append(step_record)
    return ops


def generate_plane_dirname(plane, nframes=None, frame_start=1, frame_stop=None, suffix=None):
    """Generate ``zplaneNN[_tpSTART-STOP][_suffix]`` directory name."""
    try:
        from lbm_suite2p_python.run_lsp import generate_plane_dirname as _gpd
        return _gpd(plane, nframes=nframes, frame_start=frame_start,
                    frame_stop=frame_stop, suffix=suffix)
    except ImportError:
        parts = [f"zplane{plane:02d}"]
        if nframes is not None and nframes > 1:
            stop = frame_stop if frame_stop is not None else nframes
            parts.append(f"tp{frame_start:05d}-{stop:05d}")
        if suffix:
            parts.append(suffix)
        return "_".join(parts)


def _normalize_planes(planes, num_planes: int) -> list:
    if planes is None:
        return list(range(num_planes))
    if isinstance(planes, int):
        planes = [planes]
    indices = []
    for p in planes:
        idx = p - 1
        if 0 <= idx < num_planes:
            indices.append(idx)
        else:
            print(f"Warning: plane {p} out of range (1-{num_planes}), skipping")
    return indices


def derive_tag_from_filename(path):
    name = Path(path).stem
    for tag in PIPELINE_TAGS:
        low = name.lower()
        if low.startswith(tag):
            suffix = name[len(tag):]
            if suffix and suffix[0] in ("_", "-"):
                suffix = suffix[1:]
            if suffix.isdigit():
                return f"{tag.rstrip('_')}{int(suffix)}"
    return name


def get_plane_num_from_tag(tag: str, fallback: int = None) -> int:
    import re
    match = re.search(r"(\d+)$", tag)
    return int(match.group(1)) if match else fallback


def _stat_from_A(A, dims, C=None):
    """Build a suite2p-style ``stat`` list from a CNMF spatial matrix.

    Each entry is a dict with keys used by lsp plotting / roi_stats code:
    ``ypix``, ``xpix``, ``lam``, ``npix``, ``med``, ``radius``, ``compact``,
    ``skew``. Coordinates are in full-frame (Ly, Lx) space.
    """
    from scipy import sparse
    from scipy.stats import skew as _skew

    Ly, Lx = int(dims[0]), int(dims[1])
    if sparse.issparse(A):
        A_dense = A.toarray()
    else:
        A_dense = np.asarray(A)
    n_rois = A_dense.shape[1]

    stat = []
    for i in range(n_rois):
        comp = A_dense[:, i].reshape((Ly, Lx), order="F")
        mask = comp > 0
        if not mask.any():
            ypix = np.array([0], dtype=np.int32)
            xpix = np.array([0], dtype=np.int32)
            lam = np.array([0.0], dtype=np.float32)
        else:
            ys, xs = np.where(mask)
            ypix = ys.astype(np.int32)
            xpix = xs.astype(np.int32)
            lam = comp[mask].astype(np.float32)

        npix = int(ypix.size)
        med = [int(np.median(ypix)), int(np.median(xpix))]
        radius = float(np.sqrt(npix / np.pi))
        compact = float(radius / max(1.0, npix ** 0.5))

        sk = float(_skew(C[i])) if (C is not None and C.shape[0] > i) else np.nan

        stat.append({
            "ypix": ypix,
            "xpix": xpix,
            "lam": lam,
            "npix": npix,
            "med": med,
            "radius": radius,
            "compact": compact,
            "skew": sk,
            "footprint": 0,
            "mrs": radius,
            "mrs0": radius,
            "soma_crop": np.ones(npix, dtype=bool),
            "overlap": np.zeros(npix, dtype=bool),
            "aspect_ratio": 1.0,
        })
    return np.array(stat, dtype=object)


def _iscell_from_estimates(estimates, n_total: int):
    """Build a suite2p-style ``iscell`` (n_rois, 2) array.

    Column 0 is the 0/1 accepted flag from CaImAn's
    ``idx_components``; column 1 is a probability proxy from
    ``SNR_comp`` (rescaled to [0, 1]).
    """
    iscell = np.zeros((n_total, 2), dtype=np.float32)

    accepted = getattr(estimates, "idx_components", None)
    if accepted is not None and len(accepted) > 0:
        iscell[np.asarray(accepted, dtype=int), 0] = 1.0
    else:
        iscell[:, 0] = 1.0  # accept all when evaluation didn't produce a verdict

    snr = getattr(estimates, "SNR_comp", None)
    if snr is not None and len(snr) == n_total:
        snr = np.asarray(snr, dtype=np.float32)
        snr = np.nan_to_num(snr, nan=0.0)
        smax = float(np.percentile(snr, 99)) if snr.size else 1.0
        smax = smax if smax > 0 else 1.0
        iscell[:, 1] = np.clip(snr / smax, 0.0, 1.0)
    else:
        iscell[:, 1] = iscell[:, 0]
    return iscell


def _enhanced_mean_image(mean_img):
    """Suite2p-style high-pass enhanced mean. Falls back gracefully."""
    if mean_img is None:
        return None
    try:
        from suite2p.registration import highpass_mean_image
        return highpass_mean_image(
            np.asarray(mean_img, dtype=np.float32), aspect=1.0
        )
    except Exception:
        # Manual fallback: subtract gaussian blur.
        from scipy.ndimage import gaussian_filter
        img = np.asarray(mean_img, dtype=np.float32)
        return img - gaussian_filter(img, sigma=10.0)


def _local_correlations(images):
    """Compute a Vcorr-like correlation image. Returns None on failure."""
    try:
        import caiman as cm
        from caiman.summary_images import local_correlations
        return local_correlations(images, swap_dim=False)
    except Exception:
        return None


def _convert_mmap_to_bin(mmap_path: Path, bin_path: Path) -> tuple[int, int, int]:
    """Convert a CaImAn (Lx*Ly, T) F-order float32 mmap to a suite2p
    ``(T, Ly, Lx)`` C-order int16 binary. Returns ``(T, Ly, Lx)``.
    """
    import caiman as cm
    Yr, dims, T = cm.load_memmap(str(mmap_path))
    Ly, Lx = int(dims[0]), int(dims[1])
    images = np.reshape(Yr.T, [T, Ly, Lx], order="F")
    out = np.clip(images, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    out.astype(np.int16).tofile(str(bin_path))
    return T, Ly, Lx


def _ensure_caiman_mmap(plane_dir: Path, data_bin: Path, T: int, Ly: int, Lx: int) -> Path:
    """Build the F-order (Lx*Ly, T) float32 mmap CaImAn's CNMF wants from
    a registered ``data.bin``. The caller is responsible for deleting it
    after CNMF if storage matters.
    """
    mmap_path = plane_dir / f"Yr_d1_{Ly}_d2_{Lx}_d3_1_order_C_frames_{T}_.mmap"
    if mmap_path.exists():
        return mmap_path
    src = np.memmap(str(data_bin), dtype=np.int16, mode="r", shape=(T, Ly, Lx))
    fp = np.memmap(str(mmap_path), mode="w+", dtype=np.float32,
                   shape=(Ly * Lx, T), order="F")
    for t in range(T):
        fp[:, t] = src[t].ravel(order="F").astype(np.float32)
    del fp
    return mmap_path


def pipeline(
    input_data,
    save_path: Union[str, Path] = None,
    ops: dict = None,
    planes: Union[list, int] = None,
    roi_mode: int = None,
    keep_reg: bool = True,
    keep_raw: bool = False,
    force_reg: bool = False,
    force_detect: bool = False,
    num_timepoints: int = None,
    frame_indices: list = None,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    correct_neuropil: bool = True,
    accept_all_cells: bool = False,
    save_json: bool = False,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
    rastermap_kwargs: dict = None,
    **kwargs,
) -> list:
    """Auto-detect 3D vs 4D input and dispatch to ``run_plane`` / ``run_volume``.

    Mirrors :func:`lbm_suite2p_python.pipeline` with CaImAn-specific
    extras (``force_mcorr``/``force_cnmf`` accepted as legacy aliases for
    ``force_reg``/``force_detect``).
    """
    # legacy aliases
    if kwargs.pop("force_mcorr", False):
        force_reg = True
    if kwargs.pop("force_cnmf", False):
        force_detect = True

    from mbo_utilities import imread

    reader_kwargs = reader_kwargs or {}
    writer_kwargs = dict(writer_kwargs or {})
    if num_timepoints is not None and "num_frames" not in writer_kwargs:
        writer_kwargs["num_frames"] = num_timepoints

    is_list = isinstance(input_data, (list, tuple))
    if is_list:
        is_volumetric = True
        arr = None
    else:
        if _is_lazy_array(input_data) or isinstance(input_data, np.ndarray):
            arr = input_data
        else:
            print(f"Loading input: {input_data}")
            arr = imread(_resolve_input_path(input_data), **reader_kwargs)
        is_volumetric = (_get_num_planes(arr) > 1) or (
            hasattr(arr, "ndim") and arr.ndim == 4
        )

    common = dict(
        save_path=save_path,
        ops=ops,
        keep_reg=keep_reg,
        keep_raw=keep_raw,
        force_reg=force_reg,
        force_detect=force_detect,
        frame_indices=frame_indices,
        dff_window_size=dff_window_size,
        dff_percentile=dff_percentile,
        dff_smooth_window=dff_smooth_window,
        correct_neuropil=correct_neuropil,
        accept_all_cells=accept_all_cells,
        save_json=save_json,
        rastermap_kwargs=rastermap_kwargs,
        reader_kwargs=reader_kwargs,
        writer_kwargs=writer_kwargs,
    )

    if is_volumetric:
        print("Detected 4D input, delegating to run_volume...")
        return run_volume(
            input_data=arr if arr is not None else input_data,
            planes=planes,
            **common,
        )
    print("Detected 3D input, delegating to run_plane...")
    ops_file = run_plane(input_data=arr, **common)
    return [ops_file]


def run_volume(
    input_data,
    save_path: Union[str, Path] = None,
    ops: dict = None,
    planes: Union[list, int] = None,
    keep_reg: bool = True,
    keep_raw: bool = False,
    force_reg: bool = False,
    force_detect: bool = False,
    frame_indices: list = None,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    correct_neuropil: bool = True,
    accept_all_cells: bool = False,
    save_json: bool = False,
    rastermap_kwargs: dict = None,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
    **kwargs,
) -> list:
    """Run the pipeline on a volumetric input, one plane at a time.

    Each plane gets its own ``zplaneNN`` subdirectory under ``save_path``.
    After all planes complete, volume-level stats and figures are written
    via the lsp helpers.
    """
    if kwargs.pop("force_mcorr", False):
        force_reg = True
    if kwargs.pop("force_cnmf", False):
        force_detect = True

    from mbo_utilities import imread

    reader_kwargs = reader_kwargs or {}
    writer_kwargs = writer_kwargs or {}

    input_arr = None
    input_paths = []

    if _is_lazy_array(input_data):
        input_arr = input_data
        filenames = getattr(input_arr, "filenames", None) or []
        if save_path is None and filenames:
            save_path = Path(filenames[0]).parent / "caiman_results"
    elif isinstance(input_data, np.ndarray):
        input_arr = input_data
    elif isinstance(input_data, (list, tuple)):
        input_paths = [Path(p) for p in input_data]
        if save_path is None and input_paths:
            save_path = input_paths[0].parent / "caiman_results"
    elif isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
        if save_path is None:
            save_path = input_path.parent / (input_path.stem + "_results")
        print(f"Loading volume: {input_path}")
        input_arr = imread(_resolve_input_path(input_path), **reader_kwargs)
    else:
        raise TypeError(f"Invalid input_data type: {type(input_data)}")

    if save_path is None:
        raise ValueError("save_path must be specified.")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if input_arr is not None:
        num_planes = _get_num_planes(input_arr)
    else:
        num_planes = len(input_paths)

    plane_indices = _normalize_planes(planes, num_planes)
    print(f"Processing {len(plane_indices)} planes (total: {num_planes})")
    print(f"Output: {save_path}")

    ops_files = []
    for plane_idx in plane_indices:
        plane_num = plane_idx + 1

        if input_arr is not None:
            current_input = input_arr  # run_plane extracts the plane
        else:
            current_input = input_paths[plane_idx]

        from lbm_caiman_python.postprocessing import load_ops as _load_ops
        current_ops = _load_ops(ops) if ops else default_ops()
        current_ops["plane"] = plane_num
        current_ops["num_zplanes"] = num_planes
        current_ops["nplanes"] = 1

        try:
            print(f"\n--- Plane {plane_num}/{num_planes} ---")
            ops_file = run_plane(
                input_data=current_input,
                save_path=save_path,
                ops=current_ops,
                keep_reg=keep_reg,
                keep_raw=keep_raw,
                force_reg=force_reg,
                force_detect=force_detect,
                frame_indices=frame_indices,
                dff_window_size=dff_window_size,
                dff_percentile=dff_percentile,
                dff_smooth_window=dff_smooth_window,
                correct_neuropil=correct_neuropil,
                accept_all_cells=accept_all_cells,
                save_json=save_json,
                rastermap_kwargs=rastermap_kwargs,
                reader_kwargs=reader_kwargs,
                writer_kwargs=writer_kwargs,
                _volume_plane_idx=plane_idx,
            )
            ops_files.append(ops_file)
        except Exception as e:
            print(f"ERROR processing plane {plane_num}: {e}")
            traceback.print_exc()

    if ops_files:
        _generate_volume_outputs(ops_files, save_path, rastermap_kwargs)

    return ops_files


def run_plane(
    input_data,
    save_path: Union[str, Path] = None,
    ops: dict = None,
    keep_reg: bool = True,
    keep_raw: bool = False,
    force_reg: bool = False,
    force_detect: bool = False,
    frame_indices: list = None,
    dff_window_size: int = None,
    dff_percentile: int = 20,
    dff_smooth_window: int = None,
    correct_neuropil: bool = True,
    accept_all_cells: bool = False,
    save_json: bool = False,
    rastermap_kwargs: dict = None,
    plane_name: str = None,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
    _volume_plane_idx: int = None,
    **kwargs,
) -> Path:
    """Process a single imaging plane with CaImAn and emit suite2p-format outputs."""
    if kwargs.pop("force_mcorr", False):
        force_reg = True
    if kwargs.pop("force_cnmf", False):
        force_detect = True

    from mbo_utilities import imread, imwrite
    from lbm_caiman_python.postprocessing import load_ops

    reader_kwargs = reader_kwargs or {}
    writer_kwargs = dict(writer_kwargs or {})

    input_path = None
    input_arr = None
    if _is_lazy_array(input_data):
        input_arr = input_data
        filenames = getattr(input_arr, "filenames", None) or []
        input_path = Path(filenames[0]) if filenames else Path(f"{plane_name or 'array_input'}.tif")
    elif isinstance(input_data, np.ndarray):
        input_arr = input_data
        input_path = Path(f"{plane_name or 'array_input'}.tif")
    elif isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
    else:
        raise TypeError(f"input_data must be path or array, got {type(input_data)}")

    if save_path is None:
        save_path = input_path.parent / "caiman_results"
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    ops_default = default_ops()
    ops_user = load_ops(ops) if ops else {}
    ops = {**ops_default, **ops_user}

    if "plane" not in ops:
        tag = derive_tag_from_filename(input_path)
        ops["plane"] = get_plane_num_from_tag(tag, fallback=1)
    plane = int(ops["plane"])

    if input_arr is None:
        print(f"  Loading: {input_path}")
        input_arr = imread(_resolve_input_path(input_path), **reader_kwargs)

    metadata = dict(getattr(input_arr, "metadata", {}) or {})

    if "fs" in metadata and ops.get("fs", ops.get("fr")) in (None, 30.0):
        ops["fs"] = float(metadata["fs"])
        ops["fr"] = ops["fs"]
    if "fr" in ops and "fs" not in ops:
        ops["fs"] = ops["fr"]
    if "dx" in metadata:
        ops["dx"] = float(metadata["dx"])
        ops.setdefault("dxy", (metadata.get("dy", 1.0), metadata["dx"]))
    if "dy" in metadata:
        ops["dy"] = float(metadata["dy"])
    if "dz" in metadata:
        ops["dz"] = float(metadata["dz"])
        ops.setdefault("z_step", float(metadata["dz"]))

    is_volumetric_source = _get_num_planes(input_arr) > 1

    nframes_full = None
    if hasattr(input_arr, "shape5d"):
        nframes_full = int(input_arr.shape5d[0])
    elif hasattr(input_arr, "shape"):
        nframes_full = int(input_arr.shape[0])

    subdir = plane_name if plane_name else generate_plane_dirname(
        plane=plane, nframes=nframes_full
    )
    plane_dir = save_path / subdir
    plane_dir.mkdir(exist_ok=True)
    ops_file = plane_dir / "ops.npy"

    ops["save_path"] = str(plane_dir.resolve())
    ops["ops_path"] = str(ops_file)
    ops["data_path"] = str(input_path.resolve()) if input_path.exists() else str(input_path)
    ops["source_dirname"] = plane_dir.name
    ops["source_input"] = str(input_path.name)
    ops["nplanes"] = 1
    ops["nchannels"] = 1

    print(f"  Output: {plane_dir}")

    raw_bin = plane_dir / "data_raw.bin"
    reg_bin = plane_dir / "data.bin"
    needs_raw = force_reg or not raw_bin.exists()

    if needs_raw and isinstance(input_arr, np.ndarray):
        # in-memory arrays can't be staged through mbo_utilities — fall back
        # to a direct binary dump in the same suite2p layout.
        arr3d = np.asarray(input_arr).squeeze()
        if arr3d.ndim != 3:
            raise ValueError(f"expected 3D array (T, Y, X), got {arr3d.shape}")
        T, Ly, Lx = arr3d.shape
        ops["Ly"], ops["Lx"] = Ly, Lx
        ops["nframes"] = T
        bin_start = time.time()
        print(f"  Writing data_raw.bin {arr3d.shape}...")
        np.clip(arr3d, np.iinfo(np.int16).min, np.iinfo(np.int16).max
                ).astype(np.int16).tofile(str(raw_bin))
        add_processing_step(
            ops, "binary_write",
            input_files=[str(input_path)],
            duration_seconds=time.time() - bin_start,
            extra={"plane": plane, "shape": [T, Ly, Lx]},
        )
    elif needs_raw:
        print(f"  Writing data_raw.bin...")
        bin_start = time.time()
        md_combined = {**metadata, **ops}
        write_planes = [plane] if is_volumetric_source else None
        write_kw = dict(writer_kwargs)
        if frame_indices is not None:
            write_kw["frames"] = [int(i) + 1 for i in frame_indices]
            write_kw.pop("num_frames", None)

        imwrite(
            input_arr,
            plane_dir,
            ext=".bin",
            metadata=md_combined,
            output_name="data_raw.bin",
            overwrite=True,
            planes=write_planes,
            show_progress=False,
            **write_kw,
        )
        if ops_file.exists():
            ops = np.load(ops_file, allow_pickle=True).item()
        add_processing_step(
            ops, "binary_write",
            input_files=[str(input_path)],
            duration_seconds=time.time() - bin_start,
            extra={"plane": plane, "shape": [
                ops.get("nframes", 0), ops.get("Ly", 0), ops.get("Lx", 0)
            ]},
        )

    Ly = int(ops.get("Ly", 0))
    Lx = int(ops.get("Lx", 0))
    T_raw = int(ops.get("nframes", 0))
    if not (Ly and Lx and T_raw):
        # ops written by mbo_utilities is the source of truth; fall back to
        # inferring from the binary size if the writer left them empty.
        if raw_bin.exists():
            nbytes = raw_bin.stat().st_size
            if Ly and Lx:
                T_raw = nbytes // (2 * Ly * Lx)
                ops["nframes"] = T_raw
        if not (Ly and Lx and T_raw):
            raise RuntimeError(
                "could not determine (T, Ly, Lx) for data.bin layout — "
                "ensure mbo_utilities.imwrite populated ops with Lx/Ly/nframes."
            )

    needs_reg = force_reg or not reg_bin.exists()
    do_reg = ops.get("do_motion_correction", ops.get("do_registration", 1))

    if do_reg and needs_reg:
        print("  Running motion correction...")
        mc_start = time.time()
        try:
            mc_result = _run_motion_correction(raw_bin, T_raw, Ly, Lx, ops, plane_dir)
            ops.update(mc_result)
            T_reg, Ly_reg, Lx_reg = _convert_mmap_to_bin(
                Path(mc_result["mmap_file"]), reg_bin,
            )
            ops["Ly"], ops["Lx"], ops["nframes"] = Ly_reg, Lx_reg, T_reg
            try:
                Path(mc_result["mmap_file"]).unlink(missing_ok=True)
            except Exception:
                pass
            add_processing_step(
                ops, "motion_correction",
                input_files=[str(raw_bin)],
                duration_seconds=time.time() - mc_start,
                extra={"shape": [T_reg, Ly_reg, Lx_reg]},
            )
        except Exception as e:
            print(f"  Motion correction failed: {e}")
            traceback.print_exc()
            ops["mcorr_error"] = str(e)
    elif do_reg and reg_bin.exists():
        print("  Motion correction already done.")

    # frame counts after registration
    T = int(ops.get("nframes", T_raw))
    Ly = int(ops.get("Ly", Ly))
    Lx = int(ops.get("Lx", Lx))

    # full-frame defaults so plotting helpers don't trip on missing keys
    ops.setdefault("yrange", np.array([0, Ly], dtype=np.int32))
    ops.setdefault("xrange", np.array([0, Lx], dtype=np.int32))
    ops.setdefault("badframes", np.zeros(T, dtype=bool))

    stat_file = plane_dir / "stat.npy"
    needs_detect = force_detect or not stat_file.exists()
    do_detect = ops.get("do_cnmf", ops.get("roidetect", 1))

    if do_detect and needs_detect and reg_bin.exists():
        print("  Running CNMF...")
        cnmf_start = time.time()
        try:
            mmap_file = _ensure_caiman_mmap(plane_dir, reg_bin, T, Ly, Lx)
            cnmf_result = _run_cnmf(mmap_file, ops, plane_dir, T, Ly, Lx)
            ops.update(cnmf_result["ops"])
            add_processing_step(
                ops, "cnmf",
                duration_seconds=time.time() - cnmf_start,
                extra={"n_cells": cnmf_result["n_cells"]},
            )
            try:
                mmap_file.unlink(missing_ok=True)
            except Exception:
                pass
        except Exception as e:
            print(f"  CNMF failed: {e}")
            traceback.print_exc()
            ops["cnmf_error"] = str(e)
    elif do_detect and stat_file.exists():
        print("  CNMF already done.")

    # post-processing: dff, roi stats, figures
    F_file = plane_dir / "F.npy"
    Fneu_file = plane_dir / "Fneu.npy"
    if F_file.exists() and Fneu_file.exists():
        print("  Computing dF/F...")
        dff_start = time.time()
        F = np.load(F_file)
        Fneu = np.load(Fneu_file)
        F_for_dff = F - 0.7 * Fneu if correct_neuropil else F
        try:
            from lbm_suite2p_python.postprocessing import dff_rolling_percentile
        except ImportError:
            from lbm_caiman_python.postprocessing import dff_rolling_percentile
        dff = dff_rolling_percentile(
            F_for_dff,
            window_size=dff_window_size,
            percentile=dff_percentile,
            smooth_window=dff_smooth_window,
            fs=float(ops.get("fs", ops.get("fr", 30.0))),
            tau=float(ops.get("tau", ops.get("decay_time", 1.0))),
        )
        np.save(plane_dir / "dff.npy", dff)
        add_processing_step(
            ops, "dff_calculation",
            duration_seconds=time.time() - dff_start,
            extra={"percentile": dff_percentile},
        )

    # persist post-processing knobs into ops for mbo studio
    ops["dff_window_size"] = dff_window_size
    ops["dff_percentile"] = dff_percentile
    ops["dff_smooth_window"] = dff_smooth_window
    ops["correct_neuropil"] = bool(correct_neuropil)
    ops["accept_all_cells"] = bool(accept_all_cells)
    ops["save_json"] = bool(save_json)
    ops["rastermap_kwargs"] = rastermap_kwargs

    if accept_all_cells and (plane_dir / "iscell.npy").exists():
        iscell_orig = np.load(plane_dir / "iscell.npy", allow_pickle=True)
        np.save(plane_dir / "iscell_suite2p.npy", iscell_orig.copy())
        iscell_new = iscell_orig.copy()
        iscell_new[:, 0] = 1
        np.save(plane_dir / "iscell.npy", iscell_new)

    np.save(ops_file, ops)

    # roi_stats.npy (uses lsp helper which expects all suite2p files in place)
    try:
        from lbm_suite2p_python.postprocessing import compute_roi_stats
        compute_roi_stats(plane_dir)
    except Exception as e:
        print(f"  Warning: ROI stats failed: {e}")

    # figures
    try:
        from lbm_suite2p_python.zplane import plot_zplane_figures
        plot_zplane_figures(
            plane_dir,
            dff_percentile=dff_percentile,
            dff_window_size=dff_window_size,
            dff_smooth_window=dff_smooth_window,
            correct_neuropil=correct_neuropil,
            run_rastermap=rastermap_kwargs is not None,
            rastermap_kwargs=rastermap_kwargs,
        )
    except Exception as e:
        print(f"  Warning: figure generation failed: {e}")

    if save_json:
        try:
            from lbm_suite2p_python.postprocessing import ops_to_json
            ops_to_json(ops_file)
        except Exception as e:
            print(f"  Warning: ops_to_json failed: {e}")

    # cleanup bin files per keep_raw/keep_reg
    if not keep_raw and raw_bin.exists():
        raw_bin.unlink()
    if not keep_reg and reg_bin.exists():
        reg_bin.unlink()

    return ops_file


def _run_motion_correction(raw_bin, T, Ly, Lx, ops, output_dir):
    """Run CaImAn rigid/PWrigid motion correction reading from ``data_raw.bin``.

    CaImAn's ``MotionCorrect`` accepts file paths; rather than re-encoding
    to TIFF we read the suite2p binary into a numpy memmap, save once as a
    CaImAn-format mmap, and let MotionCorrect consume that.
    """
    from caiman.motion_correction import MotionCorrect

    # stage a CaImAn-format mmap CaImAn can read directly
    in_mmap = output_dir / f"Yr_d1_{Ly}_d2_{Lx}_d3_1_order_C_frames_{T}_raw.mmap"
    if not in_mmap.exists():
        src = np.memmap(str(raw_bin), dtype=np.int16, mode="r", shape=(T, Ly, Lx))
        fp = np.memmap(str(in_mmap), mode="w+", dtype=np.float32,
                       shape=(Ly * Lx, T), order="F")
        for t in range(T):
            fp[:, t] = src[t].ravel(order="F").astype(np.float32)
        del fp

    mc = MotionCorrect(
        [str(in_mmap)],
        dview=None,
        max_shifts=ops.get("max_shifts", (6, 6)),
        strides=ops.get("strides", (48, 48)),
        overlaps=ops.get("overlaps", (24, 24)),
        max_deviation_rigid=ops.get("max_deviation_rigid", 3),
        pw_rigid=ops.get("pw_rigid", True),
        gSig_filt=ops.get("gSig_filt", (2, 2)),
        border_nan=ops.get("border_nan", "copy"),
        niter_rig=ops.get("niter_rig", 1),
        splits_rig=ops.get("splits_rig", 14),
        upsample_factor_grid=ops.get("upsample_factor_grid", 4),
    )
    mc.motion_correct(save_movie=True)

    # collect outputs in a single dict so the caller can fold them into ops
    shifts_rig = np.asarray(mc.shifts_rig) if mc.shifts_rig else np.zeros((T, 2))
    template = mc.total_template_rig
    if template is None and hasattr(mc, "total_template_els"):
        template = mc.total_template_els

    np.save(output_dir / "mcorr_shifts.npy", shifts_rig)
    if template is not None:
        np.save(output_dir / "mcorr_template.npy", template)

    mmap_out = Path(mc.mmap_file[0]) if mc.mmap_file else None
    # consolidate the mc output into the plane directory
    if mmap_out is not None and mmap_out.parent != output_dir:
        import shutil
        dst = output_dir / mmap_out.name
        if mmap_out.exists() and mmap_out != dst:
            shutil.move(str(mmap_out), str(dst))
        mmap_out = dst

    # cleanup the input-side mmap CaImAn no longer needs
    try:
        in_mmap.unlink(missing_ok=True)
    except Exception:
        pass

    result = {
        "shifts_rig": shifts_rig,
        "yoff": shifts_rig[:, 0].astype(np.float32),
        "xoff": shifts_rig[:, 1].astype(np.float32),
        "corrXY": np.ones(T, dtype=np.float32),
        "refImg": template,
        "meanImg": template if template is not None else None,
        "mmap_file": str(mmap_out) if mmap_out else None,
    }
    return result


def _run_cnmf(mmap_file, ops, output_dir, T, Ly, Lx):
    """Run CaImAn CNMF on a registered movie and emit suite2p-style files."""
    import caiman as cm
    from caiman.source_extraction.cnmf import CNMF

    Yr, dims_mm, T_mm = cm.load_memmap(str(mmap_file))
    images = np.reshape(Yr.T, [T_mm] + list(dims_mm), order="F")
    print(f"    CNMF input: shape={images.shape}")

    n_processes = ops.get("n_processes")
    if n_processes is None:
        import multiprocessing
        n_processes = max(1, multiprocessing.cpu_count() - 1)

    gnb = ops.get("gnb", ops.get("nb", 1))
    merge_thresh = ops.get("merge_thresh", ops.get("merge_thr", 0.8))
    gSig = ops.get("gSig", (4, 4))
    gSiz = ops.get("gSiz", None)

    cnmf_kwargs = dict(
        n_processes=n_processes,
        k=ops.get("K", 50),
        gSig=gSig,
        p=ops.get("p", 1),
        merge_thresh=merge_thresh,
        method_init=ops.get("method_init", "greedy_roi"),
        ssub=ops.get("ssub", 1),
        tsub=ops.get("tsub", 1),
        rf=ops.get("rf"),
        stride=ops.get("stride"),
        gnb=gnb,
        low_rank_background=ops.get("low_rank_background", True),
        update_background_components=ops.get("update_background_components", True),
        rolling_sum=ops.get("rolling_sum", True),
        only_init_patch=ops.get("only_init", False),
        normalize_init=ops.get("normalize_init", True),
        ring_size_factor=ops.get("ring_size_factor", 1.5),
        fr=float(ops.get("fr", ops.get("fs", 30.0))),
        decay_time=ops.get("decay_time", 0.4),
        min_SNR=ops.get("min_SNR", 2.5),
    )
    if gSiz is not None:
        cnmf_kwargs["gSiz"] = gSiz

    cnmf = CNMF(**cnmf_kwargs)
    cnmf.params.quality["rval_thr"] = ops.get("rval_thr", 0.85)
    cnmf.params.quality["min_cnn_thr"] = ops.get("min_cnn_thr", 0.99)
    cnmf.params.quality["use_cnn"] = ops.get("use_cnn", False)

    cnmf.fit(images)
    try:
        cnmf.estimates.evaluate_components(images, cnmf.params, dview=None)
    except Exception as e:
        print(f"    Component evaluation failed: {e}")

    est = cnmf.estimates
    n_total = est.A.shape[1] if (est.A is not None) else 0
    n_accepted = (
        len(est.idx_components) if getattr(est, "idx_components", None) is not None
        else n_total
    )
    print(f"    CNMF: {n_total} components, {n_accepted} accepted")

    # synthesize suite2p outputs
    stat = _stat_from_A(est.A, (Ly, Lx), C=est.C)
    np.save(output_dir / "stat.npy", stat)

    iscell = _iscell_from_estimates(est, n_total)
    np.save(output_dir / "iscell.npy", iscell)

    # F = raw fluorescence ≈ denoised + residual; Fneu zeros (CaImAn has no neuropil)
    F = (est.C + (est.YrA if getattr(est, "YrA", None) is not None else 0.0)).astype(np.float32)
    np.save(output_dir / "F.npy", F)
    np.save(output_dir / "Fneu.npy", np.zeros_like(F))

    spks = est.S if getattr(est, "S", None) is not None else np.zeros_like(F)
    np.save(output_dir / "spks.npy", spks.astype(np.float32))

    # registration imagery: max_proj, Vcorr, meanImgE
    max_proj = np.asarray(images).max(axis=0).astype(np.float32)
    mean_img = np.asarray(images).mean(axis=0).astype(np.float32)
    vcorr = _local_correlations(images)
    mean_e = _enhanced_mean_image(mean_img)

    ops_update = {
        "Ly": Ly,
        "Lx": Lx,
        "nframes": T,
        "max_proj": max_proj,
        "Vcorr": vcorr if vcorr is not None else max_proj,
        "meanImg": mean_img,
        "meanImgE": mean_e,
    }
    # refImg keeps the registration template when present, else falls back to mean
    if "refImg" not in ops or ops.get("refImg") is None:
        ops_update["refImg"] = mean_img

    return {
        "ops": ops_update,
        "n_cells": int(n_accepted),
    }


def _generate_volume_outputs(ops_files, save_path, rastermap_kwargs=None):
    """Write zstats.npy and volume-level figures via lsp helpers."""
    try:
        from lbm_suite2p_python.volume import (
            get_volume_stats,
            plot_volume_diagnostics,
            plot_orthoslices,
            plot_3d_roi_map,
            plot_volume_trace_figures,
        )
        from lbm_suite2p_python.zplane import plot_volume_accepted_rejected_overlay
    except ImportError as e:
        print(f"  Volume helpers unavailable: {e}")
        return

    print("\nGenerating volume statistics...")
    try:
        get_volume_stats(ops_files, overwrite=True)
    except Exception as e:
        print(f"  get_volume_stats failed: {e}")

    print("Generating volume figures...")
    for label, fn in (
        ("volume_diagnostics", lambda: plot_volume_diagnostics(ops_files, save_path)),
        ("orthoslices", lambda: plot_orthoslices(ops_files, save_path / "orthoslices.png")),
        ("3d_roi_map", lambda: plot_3d_roi_map(ops_files, save_path)),
        ("accepted_rejected", lambda: plot_volume_accepted_rejected_overlay(ops_files, save_path)),
        ("trace_figures", lambda: plot_volume_trace_figures(ops_files, save_path)),
    ):
        try:
            fn()
        except Exception as e:
            print(f"  {label} failed: {e}")
