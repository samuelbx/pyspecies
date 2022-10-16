from lib.population import Population
import numpy as np


# C^inf function which is null on negative numbers
def exp_cinf(x):
    return np.exp(-1 / np.maximum(x, 1e-18)) * np.int64(x > 0)


# Plateau function (support = [-0.5, 0.5])
def plateau(x):
    return (30 * exp_cinf(x + .5) * exp_cinf(-x + .5))**2


def MixedScenario():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: .2 + plateau(x + .1),
                     v0=lambda x: .4 - plateau(x - .1),
                     D=np.array([[1e-3, 3e-1], [3e-1, 1e-3]]))
    pop.sim(duration=10, N=1000)
    pop.sim(duration=100, N=500)
    pop.sim(duration=300, N=500)
    pop.anim()  # pop.animate(filename='mixed') to save as mp4


def LinearScenario():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: .2 + plateau(x + .1),
                     v0=lambda x: .4 - plateau(x - .1),
                     D=np.array([[1e-2, 0], [0, 1e-2]]))
    pop.sim(duration=10, N=1000)
    pop.anim()


def SineScenario1():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: np.sin(4 * x)**2,
                     v0=lambda x: np.sin(4 * x + .25)**2,
                     D=np.array([[0, 3e-1], [3e-1, 0]]))
    pop.sim(duration=.4, N=1000)
    pop.anim()


def SineScenario2():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: np.cos(2 * x)**2,
                     v0=lambda x: np.sin(6 * x)**2,
                     D=np.array([[0, 2e-1], [2e-1, 0]]))
    pop.sim(duration=.1, N=1000)
    pop.anim()


# Feel free to choose one of the previous scenarios
MixedScenario()