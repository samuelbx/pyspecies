import numpy as np


class SKT():
    def __init__(self, D: np.ndarray, R: np.ndarray):
        self.D, self.R = D, R


class LV():
    def __init__(self, alpha, beta, delta, gamma):
        assert alpha >= 0 and beta >= 0 and delta >= 0 and gamma >= 0
        self.D = np.array([[0, 0, 0], [0, 0, 0]])
        self.R = np.array([[alpha, 0, beta], [-gamma, -delta, 0]])