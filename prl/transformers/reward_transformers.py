from abc import ABC, abstractmethod
from numbers import Number

from prl.typing import HistoryABC, RewardTransformerABC, Reward
from prl.utils import timeit


class RewardTransformer(RewardTransformerABC, ABC):
    """
    Interface for classes for shaping the raw reward from wrapped environments. Object inherited
    from this class are used by the Environment class objects. Reward transformers can use all
    episode history from the beginning of the episode up to the moment of transformation.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Reward transformer UUID"""

    @abstractmethod
    def transform(self, reward: Reward, history: HistoryABC) -> Reward:
        """Transforms a reward.

        Args:
            reward: Raw reward from the wrapped environment
            history: History object

        Returns:
            Transformed reward
        """

    @abstractmethod
    def reset(self):
        """Reward transformer can be stateful, so it have to be reset after each episode."""

    @timeit
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class NoOpRewardTransformer(RewardTransformer):
    """RewardTransformer doing nothing"""

    def id(self):
        return "noop_reward_transformer"

    def transform(self, reward: Reward, history: HistoryABC) -> Number:
        return reward

    def reset(self):
        pass


class RewardShiftTransformer(RewardTransformer):
    """RewardTransformer shifting reward by some constant value"""

    def __init__(self, shift: Number):
        self.shift = shift

    def id(self):
        return "reward_shift_transformer"

    def transform(self, reward: Reward, history: HistoryABC) -> Number:
        return reward + self.shift

    def reset(self):
        pass
