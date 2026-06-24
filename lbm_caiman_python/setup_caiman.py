"""install caiman (conda-forge) and set up its data directory.

caiman is not a pip dependency: it ships compiled extensions and is only
distributed as prebuilt binaries on conda-forge. this command pulls it in
via pixi (preferred) or conda/mamba, so users never have to clone the repo
or run a separate conda install.

usage:
    lcp setup [--force] [--no-data]
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

print = partial(print, flush=True)

# caiman (conda-forge) requires numpy>=2.0 with no upper bound, so it pulls the
# latest numpy. mbo_utilities requires numpy<2.4; pin the conda numpy so it does
# not override the pypi solve and cause an unsatisfiable conflict.
NUMPY_SPEC = "numpy>=2.0,<2.4"


def _caiman_installed() -> bool:
    """return True if caiman can be imported in this environment."""
    return importlib.util.find_spec("caiman") is not None


def _pixi_project_root() -> Path | None:
    """return the pixi project root if we are running inside a pixi env."""
    manifest = os.environ.get("PIXI_PROJECT_MANIFEST")
    if manifest and Path(manifest).exists():
        return Path(manifest).parent
    root = os.environ.get("PIXI_PROJECT_ROOT")
    if root and Path(root).exists():
        return Path(root)
    return None


def _find_conda() -> str | None:
    """return a conda/mamba executable able to install conda-forge packages."""
    for name in ("mamba", "micromamba", "conda"):
        exe = shutil.which(name)
        if exe:
            return exe
    for env_var in ("MAMBA_EXE", "CONDA_EXE"):
        exe = os.environ.get(env_var)
        if exe and Path(exe).exists():
            return exe
    return None


def _run(cmd, cwd=None) -> int:
    """run a subprocess, echoing the command, and return its exit code."""
    print(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def _install_caiman() -> int:
    """install the caiman conda package via pixi or conda/mamba."""
    pixi_root = _pixi_project_root()
    if pixi_root is not None:
        pixi = shutil.which("pixi")
        if pixi is None:
            print("Inside a pixi project but the `pixi` executable was not found on PATH.")
            return 1
        print(f"Adding caiman to pixi project: {pixi_root}")
        return _run([pixi, "add", "caiman", NUMPY_SPEC], cwd=pixi_root)

    conda = _find_conda()
    if conda is not None:
        if not os.environ.get("CONDA_PREFIX"):
            print("No active conda environment (CONDA_PREFIX unset); activate one first.")
            return 1
        print(f"Installing caiman with {Path(conda).name} into {os.environ['CONDA_PREFIX']}")
        return _run([conda, "install", "-y", "-c", "conda-forge", "caiman", NUMPY_SPEC])

    print("No pixi or conda/mamba found. caiman is only distributed via conda-forge.")
    print("Install pixi (https://pixi.sh), then:")
    print("    pixi init my-lcp && cd my-lcp")
    print('    pixi add "python>=3.12.7,<3.12.10" "numpy>=2.0,<2.4"')
    print("    pixi add --pypi lbm-caiman-python")
    print("    pixi run lcp setup")
    return 1


def _setup_data() -> int:
    """populate the caiman_data directory via caimanmanager."""
    exe = shutil.which("caimanmanager")
    cmd = [exe, "install"] if exe else [sys.executable, "-m", "caiman.caimanmanager", "install"]
    return _run(cmd)


def setup_caiman(argv=None) -> int:
    """install caiman and set up its data directory. returns an exit code."""
    parser = argparse.ArgumentParser(
        prog="lcp setup",
        description="Install CaImAn (conda-forge) and set up its data directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall caiman even if it is already importable",
    )
    parser.add_argument(
        "--no-data",
        action="store_true",
        help="Skip the caimanmanager data-directory setup",
    )
    args = parser.parse_args(argv)

    if _caiman_installed() and not args.force:
        print("caiman is already installed.")
    else:
        code = _install_caiman()
        if code != 0:
            return code

    if not args.no_data:
        code = _setup_data()
        if code != 0:
            print("caimanmanager setup failed; you can retry with `caimanmanager install`.")
            return code

    if not _caiman_installed():
        print("caiman still not importable in this environment.")
        return 1

    import caiman
    print(f"caiman {getattr(caiman, '__version__', '?')} ready.")
    return 0
