import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    """
    Parses command-line arguments for rsync/robocopy CLI.
    """
    parser = argparse.ArgumentParser(
        description="CLI to sync files to HPC cluster, using rsync or robocopy as a fallback.")

    parser.add_argument(
        '--data_path',
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
        type=str,
        required=True,
        help='Path on the remote server where data should be stored (e.g., path/to/remote).'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='If specified, run rsync in dry-run mode to see what would be copied without actually copying.'
    )

    parser.add_argument(
        '--exclude',
        type=str,
        nargs='*',
        help='List of patterns to exclude from the rsync/robocopy (e.g., --exclude "*.log" "*.tmp").'
    )

    return parser.parse_args()


def check_rsync_availability():
    """
    Checks if the 'rsync' command is available on the system.
    """
    try:
        subprocess.run(['rsync', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ rsync is available.")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ rsync is not available. Falling back to robocopy (Windows only).")
        return False


def validate_paths(data_path, remote_path):
    """
    Validate that the paths exist, are correct, and not misconfigured.
    """
    # Check local data_path
    data_path = Path(data_path).resolve()  # Ensure absolute path
    if not data_path.exists():
        print(f"❌ Error: Local data_path '{data_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not data_path.is_dir():
        print(f"❌ Error: Local data_path '{data_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Check for invalid characters in remote path
    if " " in remote_path:
        print(f"❌ Error: Remote path '{remote_path}' contains spaces, which is not allowed.", file=sys.stderr)
        sys.exit(1)
    if remote_path.startswith("/"):
        print(
            f"❌ Warning: Remote path '{remote_path}' starts with a '/', it will be relative to /lustre/fs4/mbo/scratch/USER/")

    return data_path


def build_rsync_command(data_path, user, remote_path, dry_run=False, exclude_patterns=None):
    """
    Constructs the rsync command.
    """
    data_path = Path(data_path).resolve()  # Ensure absolute path
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


def build_robocopy_command(data_path, remote_path):
    """
    Constructs the robocopy command.
    """
    data_path = Path(data_path).resolve()  # Ensure absolute path
    remote_path = Path(remote_path).resolve()

    robocopy_command = [
        'robocopy',
        str(data_path),
        str(remote_path),
        '*.*',
        '/E',  # Copy all subdirectories, including empty ones
        '/Z',  # Restartable mode
        '/R:3',  # Retry 3 times on failed copies
        '/W:10'  # Wait 10 seconds between retries
    ]

    return robocopy_command


def main():
    """
    Main entry point for the script.
    """
    args = parse_args()

    print(f"📂 Validating paths...")
    validated_data_path = validate_paths(args.data_path, args.remote_path)

    # Check if rsync is available
    rsync_available = check_rsync_availability()

    if rsync_available:
        rsync_command = build_rsync_command(
            data_path=validated_data_path,
            user=args.user,
            remote_path=args.remote_path,
            dry_run=args.dry_run,
            exclude_patterns=args.exclude
        )
        print(f"🚀 Running rsync command: {' '.join(rsync_command)}")
        try:
            subprocess.run(rsync_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: rsync failed with return code {e.returncode}", file=sys.stderr)
            sys.exit(1)
    else:
        robocopy_command = build_robocopy_command(
            data_path=validated_data_path,
            remote_path=args.remote_path
        )
        print(f"🚀 Running robocopy command: {' '.join(robocopy_command)}")
        try:
            subprocess.run(robocopy_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: robocopy failed with return code {e.returncode}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
