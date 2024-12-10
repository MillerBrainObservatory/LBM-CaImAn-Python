(registration)=
# Registration

```{note}
The terms motion-correction and registration are often used interchangably.
Similary, non-rigid and peicewise-rigid are often used interchangably.
Here, peicewise-rigid registration is the **method** to correct for non-rigid motion.
```

## Overview

We use [image registration](https://en.wikipedia.org/wiki/Image_registration) to make sure that our neuron in the first frame is in the same spatial location as in frame N throughout the time-series.

As described in the {ref}`batch`, registration can be run on your data by adding an `mcorr` item to the batch with parameters:

```{code-block} python

import mesmerize_core as mc

mcorr_params = {
    'main':  # this key is necessary for specifying that these are the "main" params for the algorithm
    {
        'max_shifts': (20, 20),
        'strides': [48, 48],
        'overlaps': [24, 24],
        'max_deviation_rigid': 3,
        'border_nan': 'copy',
        'pw_rigid': True,
        'gSig_filt': (2, 2)
    },
}

mc.set_raw_data_path(path/to/tiff)
df.caiman.add_item(
    algo='mcorr',
    input_movie_path=path/to/tiff,
    params=mcorr_params,
    item_name=movie_path.stem,  # filename of the movie, but can be anything
)

```

## Registration Parameters

Parameters for motion correction are fed into `CaImAn`, someone confusingly, via the {external:func}`CNMFSetParams` function which holds and organizes parameters for registration, segmentation, deconvolution, and pre-processing steps. 

As such, you can put any parameter found in that structure into the parameters dictionary. Only the parameters that apply to registration will be used.

| **Parameter**                  | **Description**                                                                                                  |
|---------------------------------|------------------------------------------------------------------------------------------------------------------|
| `border_nan`                    | Flag for allowing NaN in the boundaries. `True` allows NaN, whereas `'copy'` copies the value of the nearest data point. |
| `gSig_filt`                     | Size of kernel for high pass spatial filtering in 1p data. If `None`, no spatial filtering is performed.         |
| `is3D`                          | Flag for 3D recordings for motion correction.                                                                    |
| `max_deviation_rigid`           | Maximum deviation in pixels between rigid shifts and shifts of individual patches.                              |
| `max_shifts`                    | Maximum shifts per dimension in pixels. (tuple of two integers)                                                |
| `min_mov`                       | Minimum value of movie. If `None`, it gets computed.                                                            |
| `niter_rig`                     | Number of iterations for rigid motion correction.                                                               |
| `nonneg_movie`                  | Flag for producing a non-negative movie.                                                                         |
| `num_frames_split`              | Split movie every `x` frames for parallel processing.                                                           |
| `overlaps`                      | Overlap between patches in pixels in pw-rigid motion correction. (tuple of two integers)                        |
| `pw_rigid`                      | Flag for performing pw-rigid motion correction.                                                                  |
| `shifts_opencv`                 | Flag for applying shifts using cubic interpolation (otherwise FFT).                                             |
| `splits_els`                    | Number of splits across time for pw-rigid registration.                                                         |
| `splits_rig`                    | Number of splits across time for rigid registration.                                                            |
| `strides`                       | How often to start a new patch in pw-rigid registration. Size of each patch will be `strides + overlaps`.       |
| `upsample_factor_grid`          | Motion field upsampling factor during FFT shifts.                                                               |
| `use_cuda`                      | Flag for using a GPU.                                                                                            |
| `indices`                       | Use that to apply motion correction only on a part of the FOV. (tuple of slices)                                |

The most important/influencial parameters in this set are:

1. {code}`gSig_filt`
: Though the description labels this parameter as applying to 1-photon calcium imaging, it is a valuable tool to handle noisy recordings in 2-photon experiments as well (TODO: link fpl example).

2. {code}`max_shift`
: Determines the maximum number of pixels that your movie will be translated in X/Y.

3. {code}`fr`
: The frame rate of our movie, which is likely different than the 30Hz default.

3. {code}`pw_rigid`
: Correct for non-uniform motion at different spatial locations in your recording.

:::{admonition} A note on `max_shift`
:class: dropdown
 
For timeseries where the FOV is sparsely labeled or a frame is corrupted, the registration process of two neighboring patches can produce very different shifts, which can lead to corrupted registered frames.
We limit the largest allowed shift with the {code}`max_shift` parameter.

:::

### gSig_filt visualization

gSig_filt is an especially useful parameter for handling noisy datasets.

To register your movie, CaImAn needs a "template", or a 2D image, to align each frame to. Applying a gaussian filter can help make the neurons have more contrast relative to the background.

Using [fastplotlib](https://github.com/fastplotlib/fastplotlib) we can easily create a visualization to gather which value of `gSig_filt` will give our neuronal landmarks the most contrast for alignmnet.

Setting `gSig_filt = (1, 1)` yields the following:

:::{figure} ../_images/gsig_1.png
:align: center
:::

Comparing the above image with `gSig_filt = (2, 2)`:

:::{figure} ../_images/gsig_2.png
:align: center
:::

... we can see much more clearly the neurons that will be used for alignment.


````{admonition} How To: Create the above visualization
:class: dropdown

```{code-block} python

## slider
from ipywidgets import IntSlider, VBox
slider_gsig_filt = IntSlider(value=3, min=1, max=33, step=1,  description="gSig_filt")

from caiman.motion_correction import high_pass_filter_space

def apply_filter(frame):
    gSig_filt = (slider_gsig_filt.value, slider_gsig_filt.value)

    # apply filter
    return high_pass_filter_space(frame, gSig_filt)

# filter shown on 2 right plots, index 1 and 2
funcs = {1:apply_filter}

iw = fpl.ImageWidget(
    data=[movie[:500], movie[:500]], # we'll apply the filter to the second movie
    frame_apply=funcs,
    figure_kwargs={"size": (1200, 600)},
    names=['raw', 'filtered'],
    cmap="gnuplot2"
)
iw.figure[0, 0].auto_scale()
iw.figure[0, 1].auto_scale()

iw.figure["filtered"].set_title(f"filtered: σ={slider_gsig_filt.value}")
iw.window_funcs = {"t": (np.mean, 3)}

def force_update(*args):
    # forces the images to update when the gSig_filt slider is moved
    iw.current_index = iw.current_index
    iw.reset_vmin_vmax()
    iw.figure["filtered"].set_title(f"filtered: σ={slider_gsig_filt.value}")

iw.reset_vmin_vmax()

slider_gsig_filt.observe(force_update, "value")

VBox([iw.show(), slider_gsig_filt])

```
````

## Registration Results

Following a registration run, results can be quickly viewed with the help of [mesmerize-viz](https://github.com/kushalkolar/mesmerize-viz).

<!-- TODO: The installation instructions wont properly install mesmerize-viz until the changes are merged -->

```{important}
Mesmerize-viz needs a separate install step, this will be fixed once the [pull request](https://github.com/kushalkolar/mesmerize-viz/pull/45) is merged

Until then:

conda activate env
git clone https://github.com/FlynnOConnell/mesmerize-viz.git
cd mesmerize-viz
pip install -e .
```

``` {code-block}

from mesmerize_viz import *

viz = df.mcorr.viz(start_index=0)
viz.show()
```

:::{figure} ../_images/mv_mcorr.png
:align: center
:::

