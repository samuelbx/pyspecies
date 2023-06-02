from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    """Abstract class for population models."""

    @abstractmethod
    def sim(
        self, X0: np.ndarray, Space: np.ndarray, Time: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        pass
