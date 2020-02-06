from abc import ABC, abstractmethod

import numpy as np

from prl.typing import FunctionApproximatorABC


class FunctionApproximator(FunctionApproximatorABC, ABC):
    """
    Class for function approximators used by the agents. For example it could
    be a neural network for value function or policy approximation.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Function Approximator UUID"""

    @abstractmethod
    def train(self, x: np.ndarray, *loss_args) -> float:
        """Trains FA for one or more steps. Returns training loss value."""

    @abstractmethod
    def predict(self, x):
        """Makes prediction based on input"""
