from pathlib import Path

import h5py
import numpy as np

from ..util.exceptions import PipelineException

def save_zstack(filename, data, metadata=None, overwrite=False):

    dims = data.shape
    if len(dims) == 4:  # volume
        imtype = 'volume'
    if len(dims) == 3:  # plane
        data = np.expand_dims(data, axis=0)
        imtype = 'plane'
    elif len(dims)== 2:  # image
        imtype = 'image'
    else:
        raise PipelineException(f'Data must be an image, plane or volume. Received shape: {dims}')

    filename = Path(filename)

    # case 1: file exists and overwrite is False
    if filename.exists() and not overwrite:
        return
    # case 2: file exists and overwrite is True
    elif filename.exists():
        filename.unlink()  # remove file
    # case 3: file does not exist, overwrite doesn't matter, create the directory
    else:
        filename.parent.mkdir(parents=True, exist_ok=True)

    filename = filename.with_suffix('.h5')
    with h5py.File(filename, 'w') as f:
        if imtype == 'volume':
            f.create_dataset('raw', data=data)

        if metadata is not None:
            for key, value in metadata.items():
                if isinstance(value, str):
                    f.attrs[key] = value.encode('utf-8')
                else:
                    f.attrs[key] = value

def load_zstack(filename):
    filename = Path(filename).with_suffix('.h5')
    with h5py.File(filename, 'r') as f:
        data = f['data'][()]
        metadata = {key: value for key, value in f.attrs.items()}
    return data, metadata

