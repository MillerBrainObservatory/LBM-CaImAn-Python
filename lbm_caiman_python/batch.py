import re as regex
from pathlib import Path
from typing import Union
import lbm_mc as mc
from contextlib import contextmanager
import pathlib

COMPUTE_BACKEND_SUBPROCESS = "subprocess"  #: subprocess backend
COMPUTE_BACKEND_SLURM = "slurm"  #: SLURM backend
COMPUTE_BACKEND_LOCAL = "local"

COMPUTE_BACKENDS = [
    COMPUTE_BACKEND_SUBPROCESS,
    COMPUTE_BACKEND_SLURM,
    COMPUTE_BACKEND_LOCAL,
]

DATAFRAME_COLUMNS = [
    "algo",
    "item_name",
    "input_movie_path",
    "params",
    "outputs",
    "added_time",
    "ran_time",
    "algo_duration",
    "comments",
    "uuid",
]


@contextmanager
def _set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


@contextmanager
def _set_windows_posix():
    """
    Set the Path class to WindowsPath on a POSIX system.
    """
    windows_backup = pathlib.WindowsPath
    try:
        pathlib.WindowsPath = pathlib.PosixPath
        yield
    finally:
        pathlib.WindowsPath = windows_backup


def load_batch(batch_path: str | Path):
    """
    Load a batch after transfering it from a Windows to a POSIX system or vice versa.

    Parameters
    ----------
    batch_path : str, Path
        The path to the batch file.

    Returns
    -------
    pandas.DataFrame
        The loaded batch.
    """
    try:
        with _set_windows_posix():
            return mc.load_batch(batch_path)
    except Exception:
        with _set_posix_windows():
            return mc.load_batch(batch_path)


def clean_batch(df):
    """
        Clean a batch of DataFrame entries by removing unsuccessful df from storage.

        This function iterates over the df of the given DataFrame, identifies
        df where the 'outputs' column is either `None` or a dictionary containing
        a 'success' key with a `False` value. For each such row, the corresponding
        item is removed using the `df.caiman.remove_item()` method, and the removal
        is saved to disk.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to be cleaned. It must have a 'uuid' column for identification
            and an 'outputs' column containing a dictionary with a 'success' key.

        Returns
        -------
        pandas.DataFrame
            The DataFrame reloaded from disk after unsuccessful items have been removed.

        Notes
        -----
        - If 'outputs' is None or does not contain 'success' as a key with a value of
          `False`, the row will be removed.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'uuid': ['123', '456', '789'],
        ...     'outputs': [{'success': True}, {'success': False}, None]
        ... })
        >>> cleaned_df = clean_batch(df)
        Removing unsuccessful batch row 1.
        Row 1 deleted.
        Removing unsuccessful batch row 2.
        Row 2 deleted.
        """
    for index, row in df.iterrows():
        # Check if 'outputs' is a dictionary and has 'success' key with value False
        if isinstance(row["outputs"], dict) and row["outputs"].get("success") is False or row["outputs"] is None:
            uuid = row["uuid"]
            print(f"Removing unsuccessful batch row {row.index}.")
            df.caiman.remove_item(uuid, remove_data=True, safe_removal=False)
            print(f"Row {row.index} deleted.")
    df.caiman.save_to_disk()
    return df.caiman.reload_from_disk()


def delete_batch_rows(df, rows_delete, remove_data=False, safe_removal=True):
    rows_delete = [rows_delete] if isinstance(rows_delete, int) else rows_delete
    uuids_delete = [row.uuid for i, row in df.iterrows() if i in rows_delete]
    for uuid in uuids_delete:
        df.caiman.remove_item(uuid, remove_data=remove_data, safe_removal=safe_removal)
    df.caiman.save_to_disk()
    return df


def validate_path(path: Union[str, Path]):
    if not regex.match("^[A-Za-z0-9@\/\\\:._-]*$", str(path)):
        raise ValueError(
            "Paths must only contain alphanumeric characters, "
            "hyphens ( - ), underscores ( _ ) or periods ( . )"
        )
    return path


def remove_batch_duplicates(df):
    """
    Remove duplicate items from a batch DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The batch DataFrame to remove duplicates from.

    Returns
    -------
    None

    """
    import hashlib
    df["hash"] = df.apply(lambda row: hashlib.sha256(row.mcorr.get_output().tobytes()).hexdigest(), axis=1)
    uuids_to_remove = []
    for _, group in df.groupby("hash"):
        if len(group) > 1:
            for idx in group.index[1:]:
                uuid = df.loc[idx, "uuid"]
                uuids_to_remove.append(uuid)
    if not uuids_to_remove:
        print("No duplicates found.")
        return
    for uuid in uuids_to_remove:
        print(f"Removing duplicate item {uuid}.")
        df.caiman.remove_item(uuid, remove_data=True, safe_removal=False)
    df.drop(columns="hash", inplace=True)
    df.caiman.save_to_disk()
    return df


def get_batch_from_path(batch_path):
    """
    Load or create a batch at the given batch_path.
    """
    try:
        df = mc.load_batch(batch_path)
        print(f"Batch found at {batch_path}")
    except (IsADirectoryError, FileNotFoundError):
        print(f"Creating batch at {batch_path}")
        df = mc.create_batch(batch_path)
    return df
