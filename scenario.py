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
                     u0=lambda x: 0.2 + plateau(x + .1),
                     v0=lambda x: 0.4 - plateau(x - .1),
                     D=np.array([[1e-3, 3e-1], [3e-1, 1e-3]]))
    pop.simulate(duration=10, N=1000)
    pop.simulate(duration=100, N=500)
    pop.simulate(duration=300, N=500)
    pop.animate()  # pop.animate(filename='mixed') to save as mp4


def LinearScenario():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: 0.2 + plateau(x + 0.1),
                     v0=lambda x: 0.4 - plateau(x - 0.1),
                     D=np.array([[1e-2, 0], [0, 1e-2]]))
    pop.simulate(duration=10, N=1000)
    pop.animate()


def SineScenario1():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: np.sin(4 * x)**2,
                     v0=lambda x: np.sin(4 * x + .25)**2,
                     D=np.array([[0, 3e-1], [3e-1, 0]]))
    pop.simulate(duration=0.4, N=1000)
    pop.animate()


def SineScenario2():
    pop = Population(Space=np.linspace(-1, 1, 1000),
                     u0=lambda x: np.cos(2 * x)**2,
                     v0=lambda x: np.sin(6 * x)**2,
                     D=np.array([[0, 2e-1], [2e-1, 0]]))
    pop.simulate(duration=0.1, N=1000)
    pop.animate()


# Feel free to choose one of the previous scenarios
MixedScenario()