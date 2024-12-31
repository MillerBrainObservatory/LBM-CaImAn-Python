import sys

from .util.io import find_files_with_extension
from .util.quality import reshape_spatial
from .batch import load_batch


def get_all_cnmf_summary(data_path, background_image="max_proj"):
    files = find_files_with_extension(data_path, '.pickle', 3)
    if not files:
        raise ValueError(f"No .pickle files found in {data_path} or its subdirectories.")
    plots = {}
    for file in files:
        try:
            df = load_batch(file)
        except Exception as e:
            print(f"Error loading {file}: {e}", file=sys.stderr)
            continue
        print(f"Loaded {file}")
        for index, row in df.iterrows():
            if isinstance(row["outputs"], dict) and row["outputs"].get("success") is False or row["outputs"] is None:
                continue
            if row['algo'] == 'cnmf':
                if background_image == 'corr':
                    bg = row.caiman.get_corr_image()
                elif background_image == 'pnr':
                    bg = row.caiman.get_pnr_image()
                elif background_image == 'max_proj':
                    bg = row.caiman.get_projection('max')
                elif background_image == 'mean_proj':
                    bg = row.caiman.get_projection('mean')
                elif background_image == 'std_proj':
                    bg = row.caiman.get_projection('std')
                elif background_image == 'all_spatial':
                    bg = reshape_spatial(row.cnmf.get_output())
                else:
                    raise ValueError(f"Background image type: {background_image} not recognized.\nMust be one of:\n"
                                     f"max_proj, min_proj, mean_proj, std_proj, pnr, or corr.")

                plots[f'{file}-{index}'] = (row.cnmf.get_contours('good'), bg)
    return plots
