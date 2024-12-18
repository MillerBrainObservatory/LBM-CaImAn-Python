(user_guide)=
# Tutorial

Below are [example notebooks](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python/tree/master/demos/notebooks) that will walk you through the full processing pipeline.

To download these notebooks:

```python
git clone https://github.com/MillerBrainObservatory/LBM-CaImAn-Python.git
cd LBM-CaImAn-Python/demos/notebooks
jupyter lab
```

See the [installation instructions](https://github.com/MillerBrainObservatory/LBM-CaImAn-Python#installation) for details on setting up the pipeline before using these notebooks.

Make sure your environment is activated via `conda activate lbm` or `source LBM-CaImAn-Python/venv/Scripts/activate` before running `jupyter lab`.

```{toctree}
---
maxdepth: 2
numbered: 3
---
notebooks/assembly.ipynb
notebooks/batch_helpers.ipynb
notebooks/motion_correction.ipynb
notebooks/segmentation.ipynb
notebooks/collation.ipynb
```
