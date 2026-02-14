"""
cli entry point for lbm-caiman-python.

usage:
    lcp <input> <output> [options]
    python -m lbm_caiman_python <input> <output> [options]
"""

import argparse
import sys
from functools import partial
from pathlib import Path

print = partial(print, flush=True)


def add_args(parser: argparse.ArgumentParser):
    """add command-line arguments to the parser."""
    from lbm_caiman_python.default_ops import default_ops

    defaults = default_ops()

    # positional arguments
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Input file or directory",
    )
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        default=None,
        help="Output directory",
    )

    # processing flags
    parser.add_argument(
        "--planes",
        type=int,
        nargs="+",
        default=None,
        help="Planes to process (1-based)",
    )
    parser.add_argument(
        "--no-mcorr",
        action="store_true",
        help="Skip motion correction",
    )
    parser.add_argument(
        "--no-cnmf",
        action="store_true",
        help="Skip CNMF",
    )
    parser.add_argument(
        "--force-mcorr",
        action="store_true",
        help="Force re-run motion correction",
    )
    parser.add_argument(
        "--force-cnmf",
        action="store_true",
        help="Force re-run CNMF",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Limit number of frames to process",
    )

    # motion correction parameters
    parser.add_argument(
        "--max-shifts",
        type=int,
        nargs=2,
        default=None,
        help=f"Maximum rigid shifts (default: {defaults['max_shifts']})",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs=2,
        default=None,
        help=f"Patch strides for piecewise-rigid (default: {defaults['strides']})",
    )
    parser.add_argument(
        "--overlaps",
        type=int,
        nargs=2,
        default=None,
        help=f"Patch overlaps (default: {defaults['overlaps']})",
    )
    parser.add_argument(
        "--gSig-filt",
        type=int,
        nargs=2,
        default=None,
        help=f"Gaussian filter size (default: {defaults['gSig_filt']})",
    )
    parser.add_argument(
        "--no-pw-rigid",
        action="store_true",
        help="Disable piecewise-rigid correction (use rigid only)",
    )

    # cnmf parameters
    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help=f"Expected number of neurons (default: {defaults['K']})",
    )
    parser.add_argument(
        "--gSig",
        type=int,
        nargs=2,
        default=None,
        help=f"Expected neuron half-width (default: {defaults['gSig']})",
    )
    parser.add_argument(
        "--min-SNR",
        type=float,
        default=None,
        help=f"Minimum SNR threshold (default: {defaults['min_SNR']})",
    )
    parser.add_argument(
        "--rval-thr",
        type=float,
        default=None,
        help=f"Correlation threshold (default: {defaults['rval_thr']})",
    )
    parser.add_argument(
        "--merge-thresh",
        type=float,
        default=None,
        help=f"Merging threshold (default: {defaults['merge_thresh']})",
    )

    # general parameters
    parser.add_argument(
        "--fr",
        type=float,
        default=None,
        help=f"Frame rate in Hz (default: {defaults['fr']})",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: auto)",
    )

    # utility flags
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    return parser


def build_ops_from_args(args) -> dict:
    """build ops dictionary from command-line arguments."""
    from lbm_caiman_python.default_ops import default_ops

    ops = default_ops()

    # map cli args to ops keys
    arg_to_ops = {
        "max_shifts": "max_shifts",
        "strides": "strides",
        "overlaps": "overlaps",
        "gSig_filt": "gSig_filt",
        "K": "K",
        "gSig": "gSig",
        "min_SNR": "min_SNR",
        "rval_thr": "rval_thr",
        "merge_thresh": "merge_thresh",
        "fr": "fr",
        "n_processes": "n_processes",
    }

    for arg_name, ops_key in arg_to_ops.items():
        value = getattr(args, arg_name.replace("-", "_"), None)
        if value is not None:
            if isinstance(value, list) and len(value) == 2:
                ops[ops_key] = tuple(value)
            else:
                ops[ops_key] = value

    # handle boolean flags
    if args.no_mcorr:
        ops["do_motion_correction"] = False
    if args.no_cnmf:
        ops["do_cnmf"] = False
    if args.no_pw_rigid:
        ops["pw_rigid"] = False

    return ops


def main():
    """main cli entry point."""
    print("\n")
    print("--- LBM-CaImAn Pipeline ---")
    print("\n")

    parser = argparse.ArgumentParser(
        description="LBM-CaImAn processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  lcp data.tiff output/
  lcp data/ results/ --planes 1 2 3
  lcp movie.tif results/ --K 100 --gSig 5 5
  lcp data.tiff output/ --no-mcorr  # skip motion correction
        """,
    )
    parser = add_args(parser)
    args = parser.parse_args()

    # handle version
    if args.version:
        import lbm_caiman_python
        print(f"lbm_caiman_python v{lbm_caiman_python.__version__}")
        return

    # check required arguments
    if args.input is None:
        parser.print_help()
        return

    # setup logging
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        print("Debug mode enabled.")

    # validate input
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # setup output
    if args.output is None:
        output_path = input_path.parent / (input_path.stem + "_caiman_results")
    else:
        output_path = Path(args.output).expanduser().resolve()

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # build ops from arguments
    ops = build_ops_from_args(args)

    # run pipeline
    from lbm_caiman_python.run_lcp import pipeline

    try:
        results = pipeline(
            input_data=str(input_path),
            save_path=str(output_path),
            ops=ops,
            planes=args.planes,
            force_mcorr=args.force_mcorr,
            force_cnmf=args.force_cnmf,
            num_timepoints=args.num_frames,
        )

        print("\n")
        print("--- Processing Complete ---")
        print(f"Results saved to: {output_path}")
        print(f"Processed {len(results)} plane(s)")

        # print summary
        for ops_path in results:
            from lbm_caiman_python.postprocessing import load_ops
            ops_result = load_ops(ops_path)
            plane = ops_result.get("plane", "?")
            n_cells = ops_result.get("n_cells", 0)
            print(f"  Plane {plane}: {n_cells} cells")

    except Exception as e:
        print(f"\nError: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
