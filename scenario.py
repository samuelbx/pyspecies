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
        [1e-3, 3e-1],
        [3e-1, 1e-3]
    ])

    # Discretization of spacetime
    K = 1000   # space
    N = 10000   # time
    Time = np.linspace(0, 2, N)
    Space = np.linspace(-1, 1, K)

    # Initial condition
    U0 = 0.2 + plateau(Space+0.1)
    V0 = 0.4 - plateau(Space-0.1)
    X0 = UVtoX(U0, V0)

    # Solve the system of PDEs using Backward Euler method
    X_list = BackwardEuler(X0, Time, Space, D)
    X2_list = BackwardEuler(X_list[-1], Time, Space, 700*D)

    # Output an animated graph
    Animate(Space, X_list, length=10)
    Animate(Space, X2_list, length=10)


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