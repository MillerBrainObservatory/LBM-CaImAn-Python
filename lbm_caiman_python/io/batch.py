#  HACK to prevent loading caiman and all of its dependencies when trying to load a batch
import re as regex
from pathlib import Path
from typing import Union
import mesmerize_core as mc

import pandas as pd

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


def clean_batch(df):
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
