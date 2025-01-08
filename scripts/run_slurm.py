import argparse
import subprocess
import time
from itertools import product
from pathlib import Path


def run_command(command, capture_output=False):
    result = subprocess.run(command, shell=True, text=True, capture_output=capture_output)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr}")
    return result.stdout if capture_output else None


def stage_data(copydir, tmpdir):
    copydir = Path(copydir).expanduser().resolve()
    print(f"Staging data from {copydir} to {tmpdir}...")
    stage_command = (
        f"mkdir -p {tmpdir} && "
        f"rsync -av --include '*/' --include '*.tif*' --exclude '*' {copydir}/ {tmpdir}/"
    )
    return stage_command


def run_mcorr(tmpdir):
    print("Running mcorr...")
    return f"lcp --batch_path {tmpdir} --run mcorr --data_path {tmpdir}"


def run_cnmf(tmpdir, gSig, K):
    print(f"Running cnmf with gSig={gSig}, K={K}...")
    return f"lcp --batch_path {tmpdir} --run cnmf --gSig {gSig} --K {K} --data_path {tmpdir}"


def transfer_results_to_remote(tmpdir, tmp_dest):
    print(f"Transferring results to remote destination {tmp_dest}...")
    transfer_command = (
        f"rsync -av -e 'ssh -i ~/.ssh/id_rsa -o IdentitiesOnly=yes' "
        f"--exclude 'plane_*' --include '*' {tmpdir}/ rbo@129.85.3.34:{tmp_dest}"
    )
    return transfer_command


def create_srun_command(tmpdir, tmp_dest, args):
    srun_prefix = (
        f"srun --mem={args.mem}G --ntasks={args.ntasks} --partition={args.partition} "
        f"--cpus-per-task={args.cpus_per_task} --time={args.time} bash -c '"
    )
    commands = [stage_data(args.copydir, tmpdir), run_mcorr(tmpdir)]
    commands.extend([run_cnmf(tmpdir, gSig, K) for gSig, K in product(args.gSig, args.K)])
    commands.append(transfer_results_to_remote(tmpdir, tmp_dest))
    return srun_prefix + " && ".join(commands) + "'"


def main():
    parser = argparse.ArgumentParser(description="Run a SLURM batch job for CNMF parameter grid search.")
    parser.add_argument("--copydir", required=True, help="Directory containing the data to process.")
    parser.add_argument("--tmp_dest", required=True, help="Remote temporary directory for rsync results.")
    parser.add_argument("--gSig", type=int, nargs="+", default=[4, 10], help="List of gSig values (default: 4 10).")
    parser.add_argument("--K", type=int, nargs="+", default=[15, 30], help="List of K values (default: 15 30).")
    parser.add_argument("--mem", type=int, default=32, help="Memory per node in GB (default: 32).")
    parser.add_argument("--ntasks", type=int, default=1, help="Number of tasks (default: 1).")
    parser.add_argument("--partition", default="hpc_a10_a", help="SLURM partition (default: hpc_a10_a).")
    parser.add_argument("--cpus_per_task", type=int, default=4, help="CPUs per task (default: 4).")
    parser.add_argument("--time", default="15:00:00", help="Time limit for the job (default: 15:00:00).")

    args = parser.parse_args()
    tmpdir = f"/tmp/data_{int(time.time())}"  # Create a unique temporary directory name

    try:
        print("Staging raw data and running computations on the compute node...")
        srun_command = create_srun_command(tmpdir, args.tmp_dest, args)
        run_command(srun_command)
    except Exception as e:
        print(f"Error occurred: {e}")
        run_command(f"rm -rf {tmpdir}")  # Cleanup in case of failure
    else:
        print("Processing completed successfully.")


if __name__ == "__main__":
    main()
