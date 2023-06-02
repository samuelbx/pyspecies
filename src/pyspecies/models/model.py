from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    """Abstract class for population models."""

    @abstractmethod
    def sim(self, Space: np.ndarray, Time: np.ndarray):
        pass
