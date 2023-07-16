# SPOT Examples

## Simple spotPython run

```python
import numpy as np
from math import inf
from spotPython.fun.objectivefunctions import analytical
from spotPython.spot import spot
# number of initial points:
ni = 7
# number of points
n = 10

fun = analytical().fun_sphere
lower = np.array([-1])
upper = np.array([1])
design_control={"init_size": ni}

spot_1 = spot.Spot(fun=fun,
            lower = lower,
            upper= upper,
            fun_evals = n,
            show_progress=True,
            design_control=design_control,)
spot_1.run()
```

## Further Examples

Examples can be found in the Hyperparameter Tuning Cookbook, e.g., [Documentation of the Sequential Parameter Optimization](https://sequential-parameter-optimization.github.io/Hyperparameter-Tuning-Cookbook/99_spot_doc.html).