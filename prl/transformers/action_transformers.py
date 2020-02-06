from abc import ABC, abstractmethod

from prl.typing import HistoryABC, ActionTransformerABC, Space, Action
from prl.utils import timeit


class ActionTransformer(ActionTransformerABC, ABC):
    """
    Interface for raw action (original actions from agent) transformers.  Object of
    this class are used by the classes implementing EnvironmentABC interface. Action
    transformers can use all episode history from the beginning of the episode up to the moment
    of transformation.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """State transformer UUID"""

    @abstractmethod
    def action_space(self, original_space: Space) -> Space:
        """

        Returns: action_space object of class gym.Space, which defines type and shape of transformed action.

        Note:
            If transformed action is from the same action_space as original
            state, then action_space is None. Information contained within action_space can
            be important for agents, so it is important to properly define an action_space.
        """

    @abstractmethod
    def transform(self, action: Action, history: HistoryABC) -> Action:
        """
        Transforms action into another representation, which must be of the form defined by
        action_space object. Input action can be in a form of numpy array, list, tuple, int, etc.

        Args:
            action: Action from the agent
            history: History object of an episode

        Returns:
            Transformed action in form defined by the action_space object.
        """

    @abstractmethod
    def reset(self):
        """Action transformer can be stateful, so it have to be reset after each episode."""

    @timeit
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class NoOpActionTransformer(ActionTransformer):
    """ActionTransformer doing nothing"""

    def action_space(self, original_space) -> Space:
        return original_space

    @property
    def id(self):
        return "noop_action_transformer"

    def transform(self, action: Action, history: HistoryABC) -> Action:
        return action

    def reset(self):
        pass
