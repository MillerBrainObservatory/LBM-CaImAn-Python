import argparse
import sys
from pathlib import Path
from fabric import Connection
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI to sync files to HPC cluster using Fabric (SFTP over SSH) with progress tracking.")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Local path to the data directory to be synced (e.g., /path/to/data).')
    parser.add_argument('--user', type=str, required=True, help='Username for the remote server (e.g., USER).')
    parser.add_argument('--remote_host', type=str, required=True,
                        help='Remote server hostname (e.g., dtn02-hpc.rockefeller.edu).')
    parser.add_argument('--remote_path', type=str, required=True,
                        help='Path on the remote server where data should be stored (e.g., /lustre/fs4/mbo/scratch/foconnell/data/).')
    parser.add_argument('--exclude', type=str, nargs='*',
                        help='List of patterns to exclude from the file transfer (e.g., --exclude "*.log" "*.tmp").')
    return parser.parse_args()


def validate_paths(data_path, remote_path):
    data_path = Path(data_path).resolve()
    if not data_path.exists():
        print(f"❌ Error: Local data_path '{data_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not data_path.is_dir():
        print(f"❌ Error: Local data_path '{data_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    if " " in remote_path:
        print(f"❌ Error: Remote path '{remote_path}' contains spaces, which is not allowed.", file=sys.stderr)
        sys.exit(1)
    if ":" in remote_path and not remote_path.startswith("/"):
        print(
            f"❌ Error: Remote path '{remote_path}' should not contain colons (:) unless specifying a remote server destination.",
            file=sys.stderr)
        sys.exit(1)

    return data_path, remote_path


def transfer_file(connection, local_file, remote_file_path, pbar):
    """
    Transfer a single file in chunks to the remote server.
    Uses Fabric's SFTP directly to track upload progress.
    """
    try:
        file_size = local_file.stat().st_size
        with open(local_file, 'rb') as file:
            with connection.sftp() as sftp:
                # Ensure the directory exists on the remote server
                remote_dir = '/'.join(remote_file_path.split('/')[:-1])
                print(f"📁 Ensuring remote directory exists: {remote_dir}")
                connection.run(f'mkdir -p "{remote_dir}"')

                # Create a new remote file and write it in chunks
                with sftp.open(remote_file_path, 'wb') as remote_file:
                    chunk_size = 1024 * 1024  # 1 MB chunk size
                    while True:
                        chunk = file.read(chunk_size)
                        if not chunk:
                            break
                        remote_file.write(chunk)
                        pbar.update(len(chunk))
    except Exception as e:
        print(f"❌ Error transferring {local_file}: {e}", file=sys.stderr)


def transfer_files(connection, local_path, remote_path, exclude_patterns=None):
    """
    Transfer files from the local path to the remote server.
    Uses Fabric's `SFTP` to transfer files via chunked uploads.
    """
    print(f"📂 Starting file transfer from {local_path} to {remote_path} on remote server.")

    all_files = [file for file in local_path.rglob('*') if file.is_file()]

    # Filter out files that match the exclude patterns
    if exclude_patterns:
        all_files = [file for file in all_files if not any(file.match(pattern) for pattern in exclude_patterns)]

    total_files = len(all_files)
    total_size = sum(file.stat().st_size for file in all_files)

    if total_files == 0:
        print(f"⚠️ No files to transfer from {local_path}. Check your exclusion patterns or local directory.")
        return

    print(f"📦 Total files to transfer: {total_files} | Total size: {total_size / 1e9:.2f} GB")

    with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Transferring files") as pbar:
        for file in all_files:
            relative_path = file.relative_to(local_path)
            remote_file_path = f"{remote_path}/{relative_path}"

            # Upload the file and track the progress
            print(f"🚀 Uploading {file} to {remote_file_path}")
            transfer_file(connection, file, remote_file_path, pbar)


def main():
    args = parse_args()

    print(f"📂 Validating paths...")
    validated_data_path, remote_path = validate_paths(args.data_path, args.remote_path)

    try:
        print(f"🔗 Connecting to {args.user}@{args.remote_host}...")
        connection = Connection(host=args.remote_host, user=args.user)
    except Exception as e:
        print(f"❌ Error: Failed to establish SSH connection to {args.remote_host} as {args.user}", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        transfer_files(
            connection=connection,
            local_path=validated_data_path,
            remote_path=remote_path,
            exclude_patterns=args.exclude
        )
    except Exception as e:
        print(f"❌ Error: Failed to transfer files to {remote_path} on {args.remote_host}", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        sys.exit(1)

    print("✅ File transfer complete!")
    connection.close()


if __name__ == "__main__":
    main()
