import numpy as np

from lib.population import Population
from lib.utils import plateau


def LotkaVolterra():
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


def MixedScenario():
    pop = Population(
        Space=np.linspace(-1, 1, 1000),
        u0=lambda x: 3 * plateau(5 * x),
        v0=lambda x: plateau(x / 2),
        D=np.array([[0, 0, 2e-2], [0, 2e-2, 0]]),
        R=np.array([[0, 0, 0], [0, 0, 0]]),
    )
    pop.sim(duration=0.1, N=100)
    pop.sim(duration=1, N=100)
    pop.sim(duration=10, N=100)
    pop.sim(duration=300, N=100)
    pop.sim(duration=2000, N=100)
    pop.anim()


def LinearScenario():
    pop = Population(
        Space=np.linspace(-1, 1, 1000),
        u0=lambda x: 0.2 + plateau(x + 0.1),
        v0=lambda x: 0.4 - plateau(x - 0.1),
        D=np.array([[1e-2, 0, 0], [1e-2, 0, 0]]),
        R=np.array([[0, 0, 0], [0, 0, 0]]),
    )
    pop.sim(duration=20, N=500)
    pop.sim(duration=80, N=500)
    pop.anim()


def SineScenario1():
    pop = Population(
        Space=np.linspace(-1, 1, 1000),
        u0=lambda x: np.sin(4 * x + 0.12) ** 2,
        v0=lambda x: np.sin(4 * x - 0.12) ** 2,
        D=np.array([[0, 0, 3e-1], [0, 3e-1, 0]]),
        R=np.array([[0, 0, 0], [0, 0, 0]]),
    )
    pop.sim(duration=0.1, N=500)
    pop.sim(duration=0.9, N=500)
    pop.anim()


def SineScenario2():
    pop = Population(
        Space=np.linspace(-1, 1, 1000),
        u0=lambda x: np.cos(2 * x) ** 2,
        v0=lambda x: np.sin(6 * x) ** 2,
        D=np.array([[0, 0, 2e-2], [0, 2e-2, 0]]),
        R=np.array([[0, 0, 0], [0, 0, 0]]),
    )
    pop.sim(duration=1, N=334)
    pop.sim(duration=10, N=333)
    pop.sim(duration=50, N=333)
    pop.anim()


def IntruderScenario():
    pop = Population(
        Space=np.linspace(-1, 1, 1000),
        u0=lambda x: 3 * plateau(5 * x),
        v0=lambda x: plateau(x / 2),
        D=np.array([[0, 0, 2e-2], [0, 2e-2, 0]]),
        R=np.array([[0, 0, 0], [0, 0, 0]]),
    )
    pop.sim(duration=0.1, N=50)
    pop.sim(duration=1, N=50)
    pop.sim(duration=10, N=50)
    pop.sim(duration=100, N=50)
    pop.sim(duration=1000, N=50)
    pop.sim(duration=10000, N=50)
    pop.anim()


LotkaVolterra()
