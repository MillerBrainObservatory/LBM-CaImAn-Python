"""
main caiman processing pipeline for lbm data.

provides pipeline(), run_volume(), and run_plane() functions that mirror
the lbm_suite2p_python api structure.
"""

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


def _get_version():
    """get package version string."""
    try:
        from lbm_caiman_python import __version__
        return __version__
    except Exception:
        return "0.0.0"


def _get_caiman_version():
    """get caiman version string."""
    try:
        import caiman
        return getattr(caiman, "__version__", "unknown")
    except ImportError:
        return "not installed"


def _is_lazy_array(obj) -> bool:
    """check if object is an mbo_utilities lazy array."""
    type_name = type(obj).__name__
    lazy_types = (
        "TiffArray", "ScanImageArray", "LBMArray", "PiezoArray",
        "Suite2pArray", "H5Array", "ZarrArray", "NumpyArray",
        "SinglePlaneArray", "ImageJHyperstackArray", "BinArray",
    )
    return type_name in lazy_types


def _get_num_planes(arr) -> int:
    """get number of z-planes from array."""
    if hasattr(arr, "num_planes"):
        return arr.num_planes
    if arr.ndim == 4:
        return arr.shape[1]
    return 1


def add_processing_step(ops, step_name, input_files=None, duration_seconds=None, extra=None):
    """
    add a processing step to ops["processing_history"].

    parameters
    ----------
    ops : dict
        the ops dictionary to update.
    step_name : str
        name of the processing step.
    input_files : list of str, optional
        list of input file paths.
    duration_seconds : float, optional
        how long this step took.
    extra : dict, optional
        additional metadata.

    returns
    -------
    dict
        the updated ops dictionary.
    """
    if "processing_history" not in ops:
        ops["processing_history"] = []

    step_record = {
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        "lbm_caiman_python_version": _get_version(),
        "caiman_version": _get_caiman_version(),
    }

    if input_files is not None:
        if isinstance(input_files, str):
            step_record["input_files"] = [input_files]
        else:
            step_record["input_files"] = [str(f) for f in input_files]

    if duration_seconds is not None:
        step_record["duration_seconds"] = round(duration_seconds, 2)

    if extra is not None:
        step_record.update(extra)

    ops["processing_history"].append(step_record)
    return ops


def generate_plane_dirname(
    plane: int,
    nframes: int = None,
    frame_start: int = 1,
    frame_stop: int = None,
    suffix: str = None,
) -> str:
    """
    generate a descriptive directory name for a plane's outputs.

    parameters
    ----------
    plane : int
        z-plane number (1-based)
    nframes : int, optional
        total number of frames.
    frame_start : int, default 1
        first frame (1-based)
    frame_stop : int, optional
        last frame (1-based).
    suffix : str, optional
        additional suffix.

    returns
    -------
    str
        directory name like "zplane01", "zplane03_tp00001-05000".
    """
    parts = [f"zplane{plane:02d}"]

    if nframes is not None and nframes > 1:
        stop = frame_stop if frame_stop is not None else nframes
        parts.append(f"tp{frame_start:05d}-{stop:05d}")

    if suffix:
        parts.append(suffix)

    return "_".join(parts)


def _normalize_planes(planes, num_planes: int) -> list:
    """
    normalize planes argument to list of 0-based indices.

    parameters
    ----------
    planes : int, list, or None
        planes to process (1-based). None means all planes.
    num_planes : int
        total number of planes available.

    returns
    -------
    list
        list of 0-based plane indices.
    """
    if planes is None:
        return list(range(num_planes))

    if isinstance(planes, int):
        planes = [planes]

    # convert 1-based to 0-based
    indices = []
    for p in planes:
        idx = p - 1
        if 0 <= idx < num_planes:
            indices.append(idx)
        else:
            print(f"Warning: plane {p} out of range (1-{num_planes}), skipping")

    return indices


def derive_tag_from_filename(path):
    """derive a folder tag from filename based on planeN, roiN patterns."""
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
    """extract the plane number from a tag string like 'plane3'."""
    import re
    match = re.search(r"(\d+)$", tag)
    if match:
        return int(match.group(1))
    return fallback


def pipeline(
    input_data,
    save_path: Union[str, Path] = None,
    ops: dict = None,
    planes: Union[list, int] = None,
    roi_mode: int = None,
    force_mcorr: bool = False,
    force_cnmf: bool = False,
    num_timepoints: int = None,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
) -> list:
    """
    unified caiman processing pipeline.

    auto-detects 3d (single plane) vs 4d (volume) input and delegates
    to run_plane or run_volume accordingly.

    parameters
    ----------
    input_data : str, Path, list, or lazy array
        input data source (file, directory, list of files, or array).
    save_path : str or Path, optional
        output directory.
    ops : dict, optional
        caiman parameters. uses default_ops() if not provided.
    planes : int or list, optional
        planes to process (1-based index).
    roi_mode : int, optional
        roi mode for scanimage data (None=stitch, 0=split, N=single).
    force_mcorr : bool, default False
        force re-run motion correction.
    force_cnmf : bool, default False
        force re-run cnmf.
    num_timepoints : int, optional
        limit number of frames to process.
    reader_kwargs : dict, optional
        arguments for mbo_utilities.imread.
    writer_kwargs : dict, optional
        arguments for writing.

    returns
    -------
    list[Path]
        list of paths to ops.npy files.
    """
    import tifffile

    reader_kwargs = reader_kwargs or {}
    writer_kwargs = writer_kwargs or {}

    if num_timepoints is not None:
        writer_kwargs["num_frames"] = num_timepoints

    # determine input type and dimensionality
    is_list = isinstance(input_data, (list, tuple))

    if is_list:
        is_volumetric = True
        arr = None
    else:
        if _is_lazy_array(input_data):
            arr = input_data
        elif isinstance(input_data, np.ndarray):
            arr = input_data
        else:
            print(f"Loading input: {input_data}")
            arr = tifffile.imread(str(input_data))

        is_volumetric = arr.ndim == 4

    # delegate to appropriate function
    if is_volumetric:
        print("Detected 4D input, delegating to run_volume...")
        input_arg = arr if arr is not None else input_data

        return run_volume(
            input_data=input_arg,
            save_path=save_path,
            ops=ops,
            planes=planes,
            force_mcorr=force_mcorr,
            force_cnmf=force_cnmf,
            reader_kwargs=reader_kwargs,
            writer_kwargs=writer_kwargs,
        )
    else:
        print("Detected 3D input, delegating to run_plane...")
        ops_path = run_plane(
            input_data=arr,
            save_path=save_path,
            ops=ops,
            force_mcorr=force_mcorr,
            force_cnmf=force_cnmf,
            reader_kwargs=reader_kwargs,
            writer_kwargs=writer_kwargs,
        )
        return [ops_path]


def run_volume(
    input_data,
    save_path: Union[str, Path] = None,
    ops: dict = None,
    planes: Union[list, int] = None,
    force_mcorr: bool = False,
    force_cnmf: bool = False,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
) -> list:
    """
    process volumetric (4d: t,z,y,x) imaging data.

    iterates over z-planes and calls run_plane for each.

    parameters
    ----------
    input_data : list, Path, or array
        input data source.
    save_path : str or Path, optional
        base directory for outputs.
    ops : dict, optional
        caiman parameters.
    planes : list or int, optional
        specific planes to process (1-based).
    force_mcorr : bool, default False
        force motion correction.
    force_cnmf : bool, default False
        force cnmf.
    reader_kwargs : dict, optional
        arguments for imread.
    writer_kwargs : dict, optional
        arguments for writing.

    returns
    -------
    list[Path]
        list of paths to ops.npy files.
    """
    import tifffile

    reader_kwargs = reader_kwargs or {}
    writer_kwargs = writer_kwargs or {}

    # handle input types
    input_arr = None
    input_paths = []

    if _is_lazy_array(input_data):
        input_arr = input_data
        if hasattr(input_arr, "filenames") and input_arr.filenames:
            if save_path is None:
                save_path = Path(input_arr.filenames[0]).parent / "caiman_results"
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
        input_arr = tifffile.imread(str(input_path))
    else:
        raise TypeError(f"Invalid input_data type: {type(input_data)}")

    if save_path is None:
        raise ValueError("save_path must be specified.")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # determine number of planes
    if input_arr is not None:
        num_planes = _get_num_planes(input_arr)
    else:
        num_planes = len(input_paths)

    # normalize planes to process
    planes_indices = _normalize_planes(planes, num_planes)

    print(f"Processing {len(planes_indices)} planes (total: {num_planes})")
    print(f"Output: {save_path}")

    ops_files = []

    for i, plane_idx in enumerate(planes_indices):
        plane_num = plane_idx + 1

        # prepare input for run_plane
        if input_arr is not None:
            # extract single plane from 4d array
            current_input = input_arr[:, plane_idx, :, :]
        else:
            if plane_idx < len(input_paths):
                current_input = input_paths[plane_idx]
            else:
                continue

        # prepare ops with plane number
        from lbm_caiman_python.postprocessing import load_ops
        current_ops = load_ops(ops) if ops else default_ops()
        current_ops["plane"] = plane_num
        current_ops["num_zplanes"] = num_planes

        try:
            print(f"\n--- Plane {plane_num}/{num_planes} ---")
            ops_file = run_plane(
                input_data=current_input,
                save_path=save_path,
                ops=current_ops,
                force_mcorr=force_mcorr,
                force_cnmf=force_cnmf,
                reader_kwargs=reader_kwargs,
                writer_kwargs=writer_kwargs,
            )
            ops_files.append(ops_file)
        except Exception as e:
            print(f"ERROR processing plane {plane_num}: {e}")
            traceback.print_exc()

    # generate volume statistics
    if ops_files:
        print("\nGenerating volume statistics...")
        try:
            _generate_volume_stats(ops_files, save_path)
        except Exception as e:
            print(f"Warning: Volume statistics failed: {e}")

    return ops_files


def run_plane(
    input_data,
    save_path: Union[str, Path] = None,
    ops: dict = None,
    force_mcorr: bool = False,
    force_cnmf: bool = False,
    plane_name: str = None,
    reader_kwargs: dict = None,
    writer_kwargs: dict = None,
) -> Path:
    """
    process a single imaging plane using caiman.

    runs motion correction and cnmf, generates diagnostic plots.

    parameters
    ----------
    input_data : str, Path, or array
        input data (file path or array).
    save_path : str or Path, optional
        output directory.
    ops : dict, optional
        caiman parameters.
    force_mcorr : bool, default False
        force motion correction.
    force_cnmf : bool, default False
        force cnmf.
    plane_name : str, optional
        custom name for output directory.
    reader_kwargs : dict, optional
        arguments for imread.
    writer_kwargs : dict, optional
        arguments for writing.

    returns
    -------
    Path
        path to ops.npy file.
    """
    import tifffile
    import caiman as cm

    reader_kwargs = reader_kwargs or {}
    writer_kwargs = writer_kwargs or {}

    # handle input type
    input_path = None
    input_arr = None

    if _is_lazy_array(input_data):
        input_arr = input_data
        filenames = getattr(input_arr, "filenames", [])
        if filenames:
            input_path = Path(filenames[0])
        elif plane_name is None:
            raise ValueError("plane_name required when input is array without filenames.")
        else:
            input_path = Path(f"{plane_name}.tif")
    elif isinstance(input_data, np.ndarray):
        input_arr = input_data
        if plane_name is None:
            plane_name = "plane01"
        input_path = Path(f"{plane_name}.tif")
    elif isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
    else:
        raise TypeError(f"input_data must be path or array, got {type(input_data)}")

    # setup save path
    if save_path is None:
        save_path = input_path.parent / "caiman_results"
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # load ops
    from lbm_caiman_python.postprocessing import load_ops
    ops_default = default_ops()
    ops_user = load_ops(ops) if ops else {}
    ops = {**ops_default, **ops_user}

    # determine plane number and directory name
    if "plane" in ops:
        plane = ops["plane"]
    else:
        tag = derive_tag_from_filename(input_path)
        plane = get_plane_num_from_tag(tag, fallback=1)
        ops["plane"] = plane

    # get metadata from input
    metadata = {}
    if input_arr is not None:
        if hasattr(input_arr, "metadata"):
            metadata = input_arr.metadata
        nframes = input_arr.shape[0] if hasattr(input_arr, "shape") else None
    else:
        print(f"  Loading: {input_path}")
        input_arr = tifffile.imread(str(input_path))
        nframes = input_arr.shape[0]

    # update ops with metadata
    if "fr" in metadata:
        ops["fr"] = metadata["fr"]
    if "dx" in metadata:
        ops["dxy"] = (metadata.get("dy", 1.0), metadata.get("dx", 1.0))

    # generate directory name
    if plane_name is not None:
        subdir_name = plane_name
    else:
        subdir_name = generate_plane_dirname(plane=plane, nframes=nframes)

    plane_dir = save_path / subdir_name
    plane_dir.mkdir(exist_ok=True)
    ops_file = plane_dir / "ops.npy"

    ops["save_path"] = str(plane_dir.resolve())
    ops["data_path"] = str(input_path.resolve()) if input_path.exists() else str(input_path)

    print(f"  Output: {plane_dir}")

    # convert to numpy array for caiman
    if hasattr(input_arr, "__array__"):
        movie_data = np.asarray(input_arr)
    else:
        movie_data = input_arr

    # save movie to temp file for caiman (requires file path)
    temp_movie_path = plane_dir / "input_movie.tif"
    if not temp_movie_path.exists() or force_mcorr:
        print("  Writing input movie...")
        import tifffile
        tifffile.imwrite(str(temp_movie_path), movie_data.astype(np.int16))

    # run motion correction
    mcorr_done = (plane_dir / "mcorr_shifts.npy").exists()
    do_mcorr = ops.get("do_motion_correction", True)

    if do_mcorr and (force_mcorr or not mcorr_done):
        print("  Running motion correction...")
        mcorr_start = time.time()
        try:
            mc_result = _run_motion_correction(temp_movie_path, ops, plane_dir)
            ops.update(mc_result)
            add_processing_step(
                ops, "motion_correction",
                input_files=[str(temp_movie_path)],
                duration_seconds=time.time() - mcorr_start,
            )
        except Exception as e:
            print(f"  Motion correction failed: {e}")
            traceback.print_exc()
            ops["mcorr_error"] = str(e)
    elif mcorr_done:
        print("  Motion correction already done, loading results...")
        ops["shifts_rig"] = np.load(plane_dir / "mcorr_shifts.npy", allow_pickle=True)

    # run cnmf
    cnmf_done = (plane_dir / "estimates.npy").exists()
    do_cnmf = ops.get("do_cnmf", True)

    if do_cnmf and (force_cnmf or not cnmf_done):
        print("  Running CNMF...")
        cnmf_start = time.time()
        try:
            # use motion-corrected movie if available
            mcorr_movie = plane_dir / "mcorr_movie.mmap"
            if mcorr_movie.exists():
                movie_for_cnmf = cm.load(str(mcorr_movie))
            else:
                movie_for_cnmf = movie_data

            cnmf_result = _run_cnmf(movie_for_cnmf, ops, plane_dir)
            ops.update(cnmf_result)
            add_processing_step(
                ops, "cnmf",
                duration_seconds=time.time() - cnmf_start,
                extra={"n_cells": cnmf_result.get("n_cells", 0)},
            )
        except Exception as e:
            print(f"  CNMF failed: {e}")
            traceback.print_exc()
            ops["cnmf_error"] = str(e)
    elif cnmf_done:
        print("  CNMF already done, loading results...")

    # save ops
    np.save(ops_file, ops)

    # generate diagnostic plots
    try:
        _generate_diagnostic_plots(plane_dir, ops)
    except Exception as e:
        print(f"  Warning: Plot generation failed: {e}")

    # cleanup temp files
    if temp_movie_path.exists() and (plane_dir / "mcorr_movie.mmap").exists():
        temp_movie_path.unlink()

    return ops_file


def _run_motion_correction(movie_path, ops, output_dir):
    """run caiman motion correction."""
    from caiman.motion_correction import MotionCorrect
    import caiman as cm

    # setup parameters
    mc = MotionCorrect(
        [str(movie_path)],
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

    # run motion correction
    mc.motion_correct(save_movie=True)

    # save results
    results = {
        "shifts_rig": mc.shifts_rig,
        "template": mc.total_template_rig,
        "mmap_file": mc.mmap_file[0] if mc.mmap_file else None,
    }

    # save shifts
    np.save(output_dir / "mcorr_shifts.npy", mc.shifts_rig)

    # save template
    if mc.total_template_rig is not None:
        np.save(output_dir / "mcorr_template.npy", mc.total_template_rig)

    # rename mmap to standard location
    if mc.mmap_file:
        mmap_src = Path(mc.mmap_file[0])
        mmap_dst = output_dir / "mcorr_movie.mmap"
        if mmap_src.exists() and mmap_src != mmap_dst:
            import shutil
            shutil.move(str(mmap_src), str(mmap_dst))
            results["mmap_file"] = str(mmap_dst)

    return results


def _run_cnmf(movie, ops, output_dir):
    """run caiman cnmf."""
    from caiman.source_extraction.cnmf import CNMF
    import caiman as cm

    # convert to caiman movie if needed
    if not isinstance(movie, cm.movie):
        movie = cm.movie(movie)

    # get dimensions
    T, Ly, Lx = movie.shape

    # setup cnmf parameters
    n_processes = ops.get("n_processes")
    if n_processes is None:
        import multiprocessing
        n_processes = max(1, multiprocessing.cpu_count() - 1)

    cnmf = CNMF(
        n_processes=n_processes,
        k=ops.get("K", 50),
        gSig=ops.get("gSig", (4, 4)),
        p=ops.get("p", 1),
        merge_thresh=ops.get("merge_thresh", 0.8),
        method_init=ops.get("method_init", "greedy_roi"),
        ssub=ops.get("ssub", 1),
        tsub=ops.get("tsub", 1),
        rf=ops.get("rf"),
        stride=ops.get("stride"),
        nb=ops.get("nb", 1),
        gnb=ops.get("gnb", 1),
        low_rank_background=ops.get("low_rank_background", True),
        update_background_components=ops.get("update_background_components", True),
        rolling_sum=ops.get("rolling_sum", True),
        only_init=ops.get("only_init", False),
        normalize_init=ops.get("normalize_init", True),
        ring_size_factor=ops.get("ring_size_factor", 1.5),
    )

    # fit cnmf
    cnmf.fit(movie)

    # evaluate components
    try:
        cnmf.estimates.evaluate_components(
            movie,
            cnmf.params,
            dview=None,
        )
    except Exception as e:
        print(f"    Component evaluation failed: {e}")

    # extract results
    estimates = cnmf.estimates
    results = {
        "n_cells": estimates.A.shape[1] if hasattr(estimates, "A") and estimates.A is not None else 0,
        "Ly": Ly,
        "Lx": Lx,
        "nframes": T,
    }

    # save estimates
    estimates_dict = {
        "A": estimates.A,
        "C": estimates.C,
        "S": estimates.S if hasattr(estimates, "S") else None,
        "b": estimates.b,
        "f": estimates.f,
        "YrA": estimates.YrA if hasattr(estimates, "YrA") else None,
        "idx_components": estimates.idx_components if hasattr(estimates, "idx_components") else None,
        "idx_components_bad": estimates.idx_components_bad if hasattr(estimates, "idx_components_bad") else None,
        "SNR_comp": estimates.SNR_comp if hasattr(estimates, "SNR_comp") else None,
        "r_values": estimates.r_values if hasattr(estimates, "r_values") else None,
    }
    np.save(output_dir / "estimates.npy", estimates_dict)

    # save fluorescence traces separately for easy access
    if estimates.C is not None:
        np.save(output_dir / "F.npy", estimates.C)
    if hasattr(estimates, "S") and estimates.S is not None:
        np.save(output_dir / "spks.npy", estimates.S)

    # compute and save dff
    try:
        from lbm_caiman_python.postprocessing import dff_rolling_percentile
        if estimates.C is not None:
            dff = dff_rolling_percentile(
                estimates.C,
                fs=ops.get("fr", 30.0),
                tau=ops.get("decay_time", 0.4),
            )
            np.save(output_dir / "dff.npy", dff)
    except Exception as e:
        print(f"    dF/F computation failed: {e}")

    return results


def _generate_diagnostic_plots(plane_dir, ops):
    """generate diagnostic plots for a processed plane."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plane_dir = Path(plane_dir)

    # load data
    estimates_file = plane_dir / "estimates.npy"
    if not estimates_file.exists():
        return

    estimates = np.load(estimates_file, allow_pickle=True).item()

    # plot 1: mean image with contours
    template_file = plane_dir / "mcorr_template.npy"
    if template_file.exists():
        template = np.load(template_file)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(template, cmap="gray")
        ax.set_title(f"Mean Image - Plane {ops.get('plane', '?')}")
        ax.axis("off")

        # overlay contours if available
        A = estimates.get("A")
        if A is not None:
            from scipy import sparse
            if sparse.issparse(A):
                A = A.toarray()

            Ly = ops.get("Ly", template.shape[0])
            Lx = ops.get("Lx", template.shape[1])

            for i in range(min(A.shape[1], 100)):
                component = A[:, i].reshape((Ly, Lx), order="F")
                ax.contour(component, levels=[component.max() * 0.5], colors="r", linewidths=0.5)

        fig.savefig(plane_dir / "01_mean_with_contours.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # plot 2: correlation image
    # (would need to compute from movie - skip for now)

    # plot 3: sample traces
    F_file = plane_dir / "F.npy"
    if F_file.exists():
        F = np.load(F_file)
        n_cells = min(10, F.shape[0])

        fig, axes = plt.subplots(n_cells, 1, figsize=(12, 2 * n_cells), sharex=True)
        if n_cells == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(F[i], "k", linewidth=0.5)
            ax.set_ylabel(f"Cell {i}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[-1].set_xlabel("Frame")
        fig.suptitle(f"Sample Traces - Plane {ops.get('plane', '?')}")
        fig.tight_layout()
        fig.savefig(plane_dir / "02_sample_traces.png", dpi=150)
        plt.close(fig)

    print(f"  Diagnostic plots saved to {plane_dir}")


def _generate_volume_stats(ops_files, save_path):
    """generate aggregate statistics for a volume."""
    stats = {
        "n_planes": len(ops_files),
        "total_cells": 0,
        "cells_per_plane": [],
        "planes": [],
    }

    for ops_file in ops_files:
        ops = np.load(ops_file, allow_pickle=True).item()
        plane = ops.get("plane", 0)
        n_cells = ops.get("n_cells", 0)

        stats["planes"].append(plane)
        stats["cells_per_plane"].append(n_cells)
        stats["total_cells"] += n_cells

    np.save(save_path / "volume_stats.npy", stats)
    print(f"  Volume stats: {stats['total_cells']} cells across {stats['n_planes']} planes")
