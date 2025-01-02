import argparse
import subprocess
import os
import shutil
import tempfile
import time
from itertools import product
from pathlib import Path


def run_command(command, capture_output=False):
    result = subprocess.run(command, shell=True, text=True, capture_output=capture_output)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr}")
    return result.stdout if capture_output else None


def main():
    parser = argparse.ArgumentParser(description="Run a SLURM batch job for CNMF parameter grid search.")
    parser.add_argument("--copydir", required=True, help="Directory containing the data to process.")
    parser.add_argument("--outdir", required=True, help="Output directory for results.")
    parser.add_argument("--gSig", type=int, nargs="+", default=[4, 10], help="List of gSig values (default: 4 10).")
    parser.add_argument("--K", type=int, nargs="+", default=[15, 30], help="List of K values (default: 15 30).")
    parser.add_argument("--mem", type=int, default=32, help="Memory per node in GB (default: 32).")
    parser.add_argument("--ntasks", type=int, default=1, help="Number of tasks (default: 1).")
    parser.add_argument("--partition", default="hpc_a10_a", help="SLURM partition (default: hpc_a10_a).")
    parser.add_argument("--cpus_per_task", type=int, default=4, help="CPUs per task (default: 4).")
    parser.add_argument("--time", default="15:00:00", help="Time limit for the job (default: 15:00:00).")

    args = parser.parse_args()
    copydir = Path(args.copydir).expanduser().resolve()
    tmpdir = f"/tmp/data_{int(time.time())}"  # Create a temporary directory name

    try:
        print("Staging raw data and running computations on the compute node...")

        # Execute rsync and subsequent commands in the same srun call
        srun_command = (
            f"srun --mem={args.mem}G --ntasks={args.ntasks} --partition={args.partition} "
            f"--cpus-per-task={args.cpus_per_task} --time={args.time} "
            f"bash -c 'mkdir -p {tmpdir} && "
            f"rsync -av --include \"*/\" --include \"*.tif*\" --exclude \"*\" {copydir}/ {tmpdir}/ && "
            f"lcp --batch_path {tmpdir} --run mcorr --data_path {tmpdir} && "
        )

        for gSig, K in product(args.gSig, args.K):
            srun_command += f"lcp --batch_path {tmpdir} --run cnmf --gSig {gSig} --K {K} --data_path {tmpdir} && "

        srun_command += f"rsync -av {tmpdir}/ {Path(args.outdir).expanduser()} && rm -rf {tmpdir}'"

        # Run the full command
        run_command(srun_command)

    except Exception as e:
        print(f"Error occurred: {e}")
        # Cleanup in case of failure
        run_command(f"rm -rf {tmpdir}")

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
