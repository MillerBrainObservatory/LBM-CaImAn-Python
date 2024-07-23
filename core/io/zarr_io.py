import tifffile
import dask.array
import scanreader


def tiffs2zarr(filenames, zarrurl, chunksize, **kwargs):
    """Write images from sequence of TIFF files as zarr."""
    with tifffile.TiffFile(filenames) as tifs:
        with tifs.aszarr() as store:
            da = dask.array.from_zarr(store)
            chunks = (chunksize,) + da.shape[1:]
            da.rechunk(chunks).to_zarr(zarrurl, **kwargs)


if __name__ == '__main__':
    import zarr
    from pathlib import Path

    root = Path().home() / 'caiman_data'
    filename = root / 'high_res.tif'
    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # parser.add_argument('filenames', nargs='+')
    # parser.add_argument('--zarrurl', required=True)
    # parser.add_argument('--chunksize', type=int, default=None)
    # args = parser.parse_args()
    reader = scanreader.read_scan(str(filename))
    for plane in range(0, reader.num_channels):
        zarr_path = root / 'zarr' / f'plane_{plane + 1}.zarr'
        print(f'Writing {zarr_path}')
        data = reader[:, :, :, plane, :].squeeze()
        zarr.save(str(zarr_path), data)
        print(f'Wrote {zarr_path} successfully!')

    # tiffs2zarr(filename, zarr_path)
    # tiffs2zarr(glob('*Ch1*.tif'), 'temp', 1000)
