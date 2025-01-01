import argparse
import subprocess
import os
import shutil
import tempfile
import time
from itertools import product


def create_temp_dir():
    return tempfile.mkdtemp(dir="/tmp", prefix="data_")


def run_command(command, capture_output=False):
    result = subprocess.run(command, shell=True, text=True, capture_output=capture_output)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr}")
    return result.stdout if capture_output else None


def transfer_results(tmpdir, job_id, dest_path):
    os.makedirs(dest_path, exist_ok=True)
    for file in [f"test_{job_id}.out", f"test_{job_id}.err"]:
        shutil.copy(os.path.join(tmpdir, file), tmpdir)

    rsync_command = (
        f"rsync -av -e 'ssh -i ~/.ssh/id_rsa -o IdentitiesOnly=yes' "
        f"--exclude 'plane_*' --include '*' {tmpdir}/ rbo@129.85.3.34:{dest_path}"
    )
    run_command(rsync_command)


def main():
    parser = argparse.ArgumentParser(description="Run CNMF grid search using SLURM.")
    parser.add_argument("--mem", type=str, required=True, help="Memory allocation (e.g., 128G).")
    parser.add_argument("--ntasks", type=int, required=True, help="Number of tasks.")
    parser.add_argument("--partition", type=str, required=True, help="SLURM partition name.")
    parser.add_argument("--cpus_per_task", type=int, required=True, help="CPUs per task.")
    parser.add_argument("--time", type=str, required=True, help="Time limit (e.g., 15:00:00).")
    parser.add_argument("--gSig", nargs="+", type=int, required=True, help="List of gSig variants.")
    parser.add_argument("--K", nargs="+", type=int, required=True, help="List of K variants.")
    parser.add_argument("--copydir", type=str, required=True, help="Path to the data directory to copy.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")

    args = parser.parse_args()

    start_time = time.time()

    # Temporary directory
    tmpdir = create_temp_dir()

    try:
        print("Staging raw data...")
        rsync_command = f"rsync -av --include '*/' --include 'plane_*' --exclude '*' {args.copydir}/ {tmpdir}/"
        run_command(rsync_command)

        print("Running mcorr...")
        mcorr_command = (
            f"srun --mem={args.mem} --ntasks={args.ntasks} --partition={args.partition} --cpus-per-task={args.cpus_per_task}"
            f" lcp --batch_path {tmpdir} --run mcorr --data_path {tmpdir}"
        )
        run_command(mcorr_command)

        print("Running cnmf with parameter grid...")
        for gSig, K in product(args.gSig, args.K):
            cnmf_command = (
                f"srun --mem={args.mem} --ntasks={args.ntasks} --partition={args.partition} --cpus-per-task={args.cpus_per_task}"
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
