# PySpecies

Ultra-fast simulation of advanced 1D population dynamics. [Theory (French)](Theory.pdf)

Renders the solution of the following system of reaction-diffusion equations:

![System of equations](videos/eq.svg)

where *u* and *v* are the concentrations of competing species.

## Quickstart
Clone this repository and start editing scenarios in `scenario.py`. For example, the following code renders a cyclic solution of the Lotka-Volterra equations:

```python
from lib.population import Population

pop = Population(
    Space=np.linspace(-1, 1, 1000),
    u0=lambda x: 1 + x * 0,
    v0=lambda x: 1 + x * 0,
    D=np.array([[0, 0, 0], [0, 0, 0]]),
    R=np.array([[1.1, 0, 0.4], [-0.1, -0.4, 0]]),
)
pop.sim(duration=20, N=500)
pop.sim(duration=100, N=500)
pop.anim()
```

where *R* and *D* contain the reaction and diffusion coefficients of the species:

![Matrices](videos/matrices.svg)
