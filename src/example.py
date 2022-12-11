import numpy as np

from pyspecies import models, pop

q = pop.Pop(
    space=(0, 1, 200),
    u0=lambda x: 1 + np.cos(2 * np.pi * x),
    v0=lambda x: 1 + np.sin(2 * np.pi * x),
    model=models.SKT(
        D=np.array([[1, 0, 1], [1e-3, 0, 0]]),
        R=np.array([[4, 2, 0], [1, 1, 0]])
    ),
)

q.sim(duration=0.1, N=200)
q.sim(duration=2.4, N=200)
q.anim()