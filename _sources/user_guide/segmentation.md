(segmentation)=
# Segmentation

## CNMF Parameters

(decay_times)=
[Decay times](https://www.janelia.org/jgcamp8-calcium-indicators)

| Protein            | Max dF/F   | Half-rise time (ms)   | Time to peak (ms)   | Half-decay time (ms)   |
|:-------------------|:-----------|:----------------------|:--------------------|:-----------------------|
| jGCaMP7f (control) | 0.21±0.1   | 24.8±6.6              | 99.5±30.2           | 181.9±76.0             |
| jGCaMP8f           | 0.41±0.12  | 7.1±0.74              | 24.8±6.1            | 67.4±11.2              |
| jGCaMP8m           | 0.76±0.22  | 7.1±0.61              | 29.0±11.2           | 118.3±13.2             |
| jGCaMP8s           | 1.11±0.22  | 10.1±0.86             | 57.0±12.9           | 306.7±32.2             |
| jGCaMP8.712*       | 0.66±0.18  | 10.9±1.24             | 41.6±8.1            | 94.8±13.3              |


``` {code-block}

from mesmerize_viz import *

viz = df.cnmf.viz(start_index=-1)
viz.show()
```

:::{figure} ../_images/mv_cnmf.png
:align: center
:::

