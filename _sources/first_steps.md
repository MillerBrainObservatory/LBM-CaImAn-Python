# First Steps

## Input/Output Directories

Each time you run an algorithm, the results are saved to disk. This is your `batch_path`.
It tracks the input file-path at the time you run the algorithm.
However, its often helpful to move results.

To allow this, we call `mc.set_parent_raw_data_path(/path/to/raw.tiff)`.
Now, our results are saved **relative to this location**.
