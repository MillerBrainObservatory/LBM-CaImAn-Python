import argparse
import glob
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


def transfer_results(tmpdir, job_id, dest_path):
    os.makedirs(dest_path, exist_ok=True)
    for file in [f"{job_id}.out", f"{job_id}.err"]:
        shutil.copy(os.path.join(tmpdir, file), tmpdir)

    rsync_command = (
        f"rsync -av -e 'ssh -i ~/.ssh/id_rsa -o IdentitiesOnly=yes' "
        f"--exclude 'plane_*' --include '*' {tmpdir}/ rbo@129.85.3.34:{dest_path}"
    )
    run_command(rsync_command)



def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def validate_tiff_files(directory):
    tiff_files = glob.glob(os.path.join(directory, '**', '*.tif'), recursive=True)
    if not tiff_files:
        raise RuntimeError(f"No TIFF files found in the directory: {directory}")
    print(f"Found {len(tiff_files)} TIFF file(s) in {directory}.")
    return tiff_files

def find_tiff_folder(directory):
    tiff_files = glob.glob(os.path.join(directory, '**', '*.tif'), recursive=True)
    if not tiff_files:
        raise RuntimeError(f"No TIFF files found in the directory: {directory}")
    tiff_dirs = {os.path.dirname(tiff) for tiff in tiff_files}
    if len(tiff_dirs) > 1:
        raise RuntimeError(f"Multiple folders containing TIFF files found: {tiff_dirs}")
    return tiff_dirs.pop()


def main():
    parser = argparse.ArgumentParser(description="Run a SLURM batch job for CNMF parameter grid search.")

    # Required arguments
    parser.add_argument("--copydir", required=True, help="Directory containing the data to process.")
    parser.add_argument("--outdir", required=True, help="Output directory for results.")

    # Optional SLURM configuration arguments
    parser.add_argument("--mem", type=int, default=32, help="Memory per node in GB (default: 32).")
    parser.add_argument("--ntasks", type=int, default=1, help="Number of tasks (default: 1).")
    parser.add_argument("--partition", default="hpc_a10_a", help="SLURM partition (default: hpc_a10_a).")
    parser.add_argument("--cpus_per_task", type=int, default=4, help="CPUs per task (default: 4).")
    parser.add_argument("--time", default="15:00:00", help="Time limit for the job (default: 15:00:00).")

    # Parameter grid arguments
    parser.add_argument("--gSig", type=int, nargs="+", default=[4, 10], help="List of gSig values (default: 4 10).")
    parser.add_argument("--K", type=int, nargs="+", default=[15, 30], help="List of K values (default: 15 30).")

    args = parser.parse_args()
    start_time = time.time()

    # Temporary directory
    tmpdir = tempfile.mkdtemp()

    try:
        print("Staging raw data...")
        copydir = os.path.expanduser(args.copydir)
        rsync_command = f"rsync -av --include '*/' --include 'plane_*' --exclude '*' {copydir} {tmpdir}"

        run_command(rsync_command)

        print("Running mcorr...")
        mcorr_command = (
            f"srun --mem={args.mem}G --ntasks={args.ntasks} --partition={args.partition} --cpus-per-task={args.cpus_per_task}"
            f" lcp --batch_path {tmpdir} --run mcorr --data_path {Path(tmpdir)}/{copydir.name}"
        )
        run_command(mcorr_command)

        print("Running cnmf with parameter grid...")
        for gSig, K in product(args.gSig, args.K):
            cnmf_command = (
                f"srun --mem={args.mem}G --ntasks={args.ntasks} --partition={args.partition} --cpus-per-task={args.cpus_per_task}"
                f" lcp --batch_path {tmpdir} --run cnmf --gSig {gSig} --K {K} --data_path 0"
            )
            run_command(cnmf_command)

        print("Transferring results...")
        dest_path = os.path.expanduser(os.path.join(args.outdir, os.environ.get("SLURM_JOB_ID", "unknown_job")))
        transfer_results(tmpdir, os.environ.get("SLURM_JOB_ID", "unknown_job"), dest_path)

    finally:
        print("Cleaning up...")
        shutil.rmtree(tmpdir)

    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Execution Time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")


if __name__ == "__main__":
    main()
