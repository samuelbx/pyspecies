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
    q.sim(duration=2 * 10**i, N=100)

# Animate the result
q.anim()

# Show the evolution of the population over space and time
# q.heatmap()

# Show the final state of the population (100%)
# q.snapshot(1)
