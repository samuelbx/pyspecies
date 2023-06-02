# PySpecies

[![PyPI version](https://badge.fury.io/py/pyspecies.svg)](https://badge.fury.io/py/pyspecies)

Blazing-fast simulation of population dynamics, based on the Shigesada-Kawasaki-Teramoto (SKT) reaction-diffusion model. [[PubMed '79]](https://pubmed.ncbi.nlm.nih.gov/513804/)

Supports stochastic approximate resolution using the Gillespie algorithm, and adding custom population models. Uses periodic boundary conditions.

![Population dynamics simulation](https://github.com/samuelbx/pyspecies/raw/main/misc/example.gif)

## Installation

```bash
pip install pyspecies
```

## Usage

For example, the following code computes a solution of the SKT model and converges to a non-homogeneous steady state:

```python
import numpy as np

from pyspecies import models, pop

# Define population and interaction model
q = pop.Pop(
    space=(0, 1, 200),
    u0=lambda x: 1 + np.cos(2 * np.pi * x),
    v0=lambda x: 1 + np.sin(2 * np.pi * x),
    model=models.SKT(
        D=np.array([[5e-3, 0, 3], [5e-3, 0, 0]]),
        R=np.array([[5, 3, 1], [2, 1, 3]])
    ),
)

# Simulate with increasing speeds
for i in range(-2, 2):
    q.sim(duration=2*10**i, N=100)

# Animate the result
q.anim()

# Show the evolution of the population over space and time
# q.heatmap()

# Show the final state of the population (100%)
# q.snapshot(1)
```

See [[Breden '19]](https://arxiv.org/pdf/1910.03436.pdf) at page 2 for a description of the SKT model and its parameters `D` and `R`.

This code displays a cyclic, homogenous solution of the Lotka-Volterra equations:

```python
p = pop.Pop(
    space = (0, 1, 10),
    u0 = lambda x: 1 + 0*x,  # IC for prey
    v0 = lambda x: 1 + 0*x,  # IC for predator
    model = models.LV(1.1, 0.4, 0.4, 0.1)
)

p.sim(duration=20, N=200)
p.sim(duration=100, N=200)
p.anim()
```

Feel free to experiment the discrete model `models.Discrete` to see an approximate resolution using the Gillespie algorithm.

Models can be defined by the user, for example:

```python
class CustomModel(models.Model):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sim(self, X0, Space, Time):
        # Space and time evolution of the population
        Xlist, Tlist = None, None
        # ...
        return Xlist, Tlist
```