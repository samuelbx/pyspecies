from lib.animation import Animate
from lib.euler import BackwardEuler
from lib.utils import UVtoX
import numpy as np


# C^inf function which is null on negative numbers
def exp_cinf(x):
    return np.exp(-1 / np.maximum(x, 1e-18)) * np.int64(x > 0)


# Plateau function (support = [-0.5, 0.5])
def plateau(x):
    return (30 * exp_cinf(x + 0.5) * exp_cinf(-x + 0.5))**2


def CrossDiffusionScenario1():
    # PDE coefficients
    D = np.array([
        [1e-20, 3e-4],
        [3e-4, 1e-20]
    ])

    # Discretization of spacetime
    K = 1000   # space
    N = 5000   # time
    Time = np.linspace(0, 15, N)
    Space = np.linspace(-0.6, 0.6, K)

    # Initial condition
    U0 = 0.2 + plateau(Space+0.1)
    V0 = 0.4 - plateau(Space-0.1)
    X0 = UVtoX(U0, V0)

    # Solve the system of PDEs using Backward Euler method
    X_list = BackwardEuler(X0, Time, Space, D)

    # Output an animated graph
    Animate(Space, X_list, length=10)


def LinearDiffusionScenario1():
    # PDE coefficients
    D = np.array([
        [1e-2, 6e-40],
        [6e-40, 1e-2]
    ])

    # Discretization of spacetime
    K = 1000   # space
    N = 4000   # time
    Time = np.linspace(0, 10, N)
    Space = np.linspace(-0.6, 0.6, K)

    # Initial condition
    U0 = 0.2 + plateau(Space+0.1)
    V0 = 0.4 - plateau(Space-0.1)
    X0 = UVtoX(U0, V0)

    # Solve the system of PDEs using Backward Euler method
    X_list = BackwardEuler(X0, Time, Space, D)

    # Output an animated graph
    Animate(Space, X_list, length=10)


CrossDiffusionScenario1()
# LinearDiffusionScenario1()    # Uncomment to see this scenario