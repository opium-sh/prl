from abc import ABC, abstractmethod

import numpy as np

from prl.typing import HistoryABC, StateTransformerABC, State
from prl.utils import timeit


class StateTransformer(StateTransformerABC, ABC):
    """
    Interface for raw states (original states from wrapped environments) transformers.  Object of
    this class are used by the classes implementing EnvironmentABC interface. State
    transformers can use all episode history from the beginning of the episode up to the moment
    of transformation.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """State transformer UUID"""

    @abstractmethod
    def transform(self, state: State, history: HistoryABC) -> State:
        """
        Transforms observed state into another representation, which must be of the form defined by
        observation_space object. Input state must be in a form of numpy.ndarray.

        Args:
            state: State from wrapped environment
            history: History object

        Returns:
            Transformed state in form defined by the observation_space object.
        """

    @abstractmethod
    def reset(self):
        """State transformer can be stateful, so it have to be reset after each episode."""

    @timeit
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class NoOpStateTransformer(StateTransformer):
    """StateTransformer doing nothing"""

    @property
    def id(self):
        return "noop_state_transformer"

    def transform(self, state: State, history: HistoryABC) -> State:
        return state

    def reset(self):
        pass


class StateShiftTransformer(StateTransformer):
    """StateTransformer shifting reward by some constant vector"""

    def __init__(self, shift_tensor: np.ndarray):
        self.shift_tensor = shift_tensor

    @property
    def id(self):
        return "state_shift_transformer"

    def transform(self, state: State, history: HistoryABC) -> State:
        return state + self.shift_tensor

    def reset(self):
        pass


class PongTransformer(StateTransformer):
    """StateTransformer for Pong atari game"""

    def __init__(
        self, resize_factor: int = 2, crop: bool = True, flatten: bool = False
    ):
        self.resize_factor = resize_factor
        self.crop = crop
        self.flatten = flatten
        self._prev_obs = None

    def _transform(self, x):
        if self.crop:
            x = x[35:195]  # crop
        x = x[
            :: self.resize_factor, :: self.resize_factor, 0
        ]  # downsample by factor of 2
        x[x == 144] = 0  # erase background (background type 1)
        x[x == 109] = 0  # erase background (background type 2)
        x[x != 0] = 1  # everything else (paddles, ball) just set to 1
        if self.flatten:
            x = x.ravel()
        else:
            x = np.expand_dims(x, -1)
        return x.astype(np.float32)

    @property
    def id(self):
        return "pong_transformer"

    def transform(self, observation: State, history: HistoryABC) -> State:
        observation = self._transform(observation)
        diff = observation
        if self._prev_obs is not None:
            diff -= self._prev_obs
        self._prev_obs = observation
        return diff

    def reset(self):
        self._prev_obs = None
