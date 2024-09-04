import os


def lbm_iter(file_name: [os.PathLike], subindices, var_name_hdf5, outtype):
    if isinstance(file_name, list) or isinstance(file_name, tuple):
        for fname in file_name:
            iterator = lbm_iter(fname, subindices, var_name_hdf5, outtype)
            while True:
                try:
                    yield next(iterator)
                except:
                    break
