import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    """
    Parses command-line arguments for rsync.
    """
    parser = argparse.ArgumentParser(description="CLI to run rsync for copying files to HPC cluster.")

    parser.add_argument(
        '--data_path',
        '--data-path',
        type=str,
        required=True,
        help='Local path to the data directory to be synced (e.g., /path/to/data).'
    )

    parser.add_argument(
        '--user',
        type=str,
        required=True,
        help='Username for the remote server (e.g., USER).'
    )

    parser.add_argument(
        '--remote_path',
        '--remote-path',
        type=str,
        required=True,
        help='Path on the remote server where data should be stored (e.g., path/to/remote).'
    )

    parser.add_argument(
        '--dry_run',
        '--dry-run',
        action='store_true',
        help='If specified, run rsync in dry-run mode to see what would be copied without actually copying.'
    )

    parser.add_argument(
        '--exclude',
        type=str,
        nargs='*',
        help='List of patterns to exclude from the rsync (e.g., --exclude "*.log" "*.tmp").'
    )

    return parser.parse_args()


def build_rsync_command(data_path, user, remote_path, dry_run=False, exclude_patterns=None):
    """
    Constructs the rsync command.
    """
    data_path = Path(data_path).resolve()  # Ensure absolute path
    if not data_path.exists():
        print(f"Error: data_path '{data_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not data_path.is_dir():
        print(f"Error: data_path '{data_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    remote_full_path = f"{user}@dtn02-hpc.rockefeller.edu:/lustre/fs4/mbo/scratch/{user}/{remote_path}"

    rsync_command = [
        'rsync',
        '-avPh',
        f'{data_path}/*',
        remote_full_path
    ]

    if dry_run:
        rsync_command.append('--dry-run')

    if exclude_patterns:
        for pattern in exclude_patterns:
            rsync_command.extend(['--exclude', pattern])

    return rsync_command


def main():
    """
    Main entry point for the script.
    """
    args = parse_args()

    rsync_command = build_rsync_command(
        data_path=args.data_path,
        user=args.user,
        remote_path=args.remote_path,
        dry_run=args.dry_run,
        exclude_patterns=args.exclude
    )

    print(f"Running rsync command: {' '.join(rsync_command)}")

    try:
        subprocess.run(rsync_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: rsync failed with return code {e.returncode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
