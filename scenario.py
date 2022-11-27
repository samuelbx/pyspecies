import numpy as np

from lib.population import Population
from lib.utils import plateau


def Test():
    pop = Population(
        Space=np.linspace(-1, 1, 1000),
        u0=lambda x: 0.2 + plateau(x + 0.1),
        v0=lambda x: 0.4 - plateau(x - 0.1),
        D=np.array([[1, 1, 1], [1, 1, 1]]),
        R=np.array([[1, 1, 1], [1, 1, 1]]),
    )
    pop.sim(duration=1, N=50)
    pop.anim()


# TODO: adapt scenarios to the new spec
def MixedScenario():
    pop = Population(
        Space=np.linspace(-1, 1, 1000),
        u0=lambda x: 0.2 + plateau(x + 0.1),
        v0=lambda x: 0.4 - plateau(x - 0.1),
        D=np.array([[1e-3, 3e-1], [3e-1, 1e-3]]),
    )
    pop.sim(duration=1, N=250)
    pop.sim(duration=10, N=250)
    pop.sim(duration=100, N=250)
    pop.sim(duration=1000, N=250)
    pop.anim()
    # pop.anim(filename='mixed') to save as mp4


def LinearScenario():
    pop = Population(
        Space=np.linspace(-1, 1, 1000),
        u0=lambda x: 0.2 + plateau(x + 0.1),
        v0=lambda x: 0.4 - plateau(x - 0.1),
        D=np.array([[1e-2, 0, 0], [1e-2, 0, 0]]),
    )
    pop.sim(duration=10, N=1000)
    pop.anim()


def SineScenario1():
    pop = Population(
        Space=np.linspace(-1, 1, 1000),
        u0=lambda x: np.sin(4 * x + 0.12) ** 2,
        v0=lambda x: np.sin(4 * x - 0.12) ** 2,
        D=np.array([[0, 3e-1], [3e-1, 0]]),
    )
    pop.sim(duration=0.1, N=500)
    pop.sim(duration=0.9, N=500)
    pop.anim()


def SineScenario2():
    pop = Population(
        Space=np.linspace(-1, 1, 1000),
        u0=lambda x: np.cos(2 * x) ** 2,
        v0=lambda x: np.sin(6 * x) ** 2,
        D=np.array([[0, 2e-2], [2e-2, 0]]),
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
        D=np.array([[0, 2e-2], [2e-2, 0]]),
    )
    pop.sim(duration=0.1, N=50)
    pop.sim(duration=1, N=50)
    pop.sim(duration=10, N=50)
    pop.sim(duration=100, N=50)
    pop.sim(duration=1000, N=50)
    pop.sim(duration=10000, N=50)
    pop.anim()


Test()
