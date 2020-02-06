import pickle
import time
from abc import ABC
from typing import Tuple, Dict

import gym
import numpy as np

from prl.storage import History
from prl.transformers.action_transformers import (
    NoOpActionTransformer,
    ActionTransformer,
)
from prl.transformers.reward_transformers import (
    NoOpRewardTransformer,
    RewardTransformer,
)
from prl.transformers.state_transformers import NoOpStateTransformer, StateTransformer
from prl.typing import (
    StateTransformerABC,
    RewardTransformerABC,
    ActionTransformerABC,
    Space,
    EnvironmentABC,
    HistoryABC,
    State,
    Action,
    Reward,
)
from prl.utils import timeit


class Environment(EnvironmentABC, ABC):
    """
    Interface for wrappers for gym-like environments. It can use :py:class:`~prl.transformers.state_transformers.StateTransformer` and
    :py:class:`~prl.transformers.reward_transformers.RewardTransformer` to shape states and rewards to a convenient form for the agent. It can also
    use :py:class:`~prl.transformers.action_transformers.ActionTransformer` to change representation from the suitable to the agent to the required
    by the environments.

    Environment also keeps the history of current episode, so it doesn't have to be implemented
    on the agent side. All the transformers can use this history to transform states, actions
    and rewards.


    Args:
        env: Environment with gym like API
        environment_id: ID of the env
        state_transformer: Object of the class :py:class:`~prl.transformers.state_transformers.StateTransformer`
        reward_transformer: Object of the class :py:class:`~prl.transformers.reward_transformers.RewardTransformer`
        action_transformer: Object of the class :py:class:`~prl.transformers.action_transformers.ActionTransformer`
    """

    def __init__(
        self,
        env: gym.Env,
        environment_id: str = "Environment_wrapper",
        state_transformer: StateTransformerABC = NoOpStateTransformer(),
        reward_transformer: RewardTransformerABC = NoOpRewardTransformer(),
        action_transformer: ActionTransformerABC = NoOpActionTransformer(),
        expected_episode_length: int = 512,
        dump_history: bool = False,
    ):
        self._id = environment_id
        self._state_transformer = state_transformer
        self._reward_transformer = reward_transformer
        self._action_transformer = action_transformer
        self._state_history = None
        self._env = env
        self._action_dtype = None
        self.true_reward = False
        self.initial_history_length = 2 * expected_episode_length
        self._dump_history = dump_history

    @property
    def id(self):
        """Environment UUID"""
        return self._id

    @property
    def state_transformer(self) -> StateTransformer:
        """
        StateTransformer object for state transformations. It can be used for changing
        representation of the state. For example it can be used for simply subtracting constant
        vector from  the state, stacking the last N states or transforming image into compressed
        representation using autoencoder.

        Returns:
            :py:class:`~prl.transformers.state_transformers.StateTransformer` object
        """

        return self._state_transformer

    @property
    def reward_transformer(self) -> RewardTransformerABC:
        """
        Reward transformer object for reward shaping like taking the sign of the original reward
        or adding reward for staying on track in a car racing game.

        Returns:
            :py:class:`~prl.transformers.reward_transformers.RewardTransformer` object
        """
        return self._reward_transformer

    @property
    def action_transformer(self) -> ActionTransformerABC:
        """
        Action transformers can be used to change the representation of actions like changing the
        coordinate system or feeding only a difference from the last action for continuous action
        space. ActionTransformer is used to change representation from the suitable to the agent
        to the required by the wrapped environments.

        Returns:
            :py:class:`~prl.transformers.action_transformers.ActionTransformer` object
        """
        return self._action_transformer

    @property
    def action_space(self) -> Space:
        """

        Returns:
            action_space object from the :py:attr:`action_transformer`

        """
        return self._action_transformer.action_space(self._env.action_space)

    @property
    def observation_space(self) -> Space:
        """

        Returns:
            observation_space object from the :py:attr:`state_transformer`
        """
        transformed_state = self._state_transformer(
            self._env.observation_space.sample(),
            History(self._env.observation_space.sample(), np.int32, 2),
        )

        return TransformedSpace(
            shape=transformed_state.shape,
            dtype=transformed_state.dtype,
            transformed_state=transformed_state,
        )

    @property
    def state_history(self) -> HistoryABC:
        """

        Returns:
            Current episode history
        """
        return self._state_history

    @timeit
    def reset(self) -> State:
        """Resets the environments to initial state and returns this initial state.

        Returns:
            New state
        """
        if self._dump_history and self._state_history:
            fname = "history_" + self.id + "_" + str(time.time()) + ".pkl"
            with open(fname, "wb") as file:
                pickle.dump(self.state_history, file)
        initial_state = self._env.reset()
        self.state_transformer.reset()
        self.reward_transformer.reset()
        self.action_transformer.reset()

        sampled_action = self.action_space.sample()
        if isinstance(sampled_action, int):
            self._action_dtype = np.int32
        else:
            raise NotImplementedError(
                "Only implemented for Discrete spaces with one action to take."
            )
        self._state_history = History(
            initial_state,
            self._action_dtype,
            initial_length=self.initial_history_length,
        )
        return self.state_transformer(initial_state, self._state_history)

    def step(self, action: Action) -> Tuple[State, Reward, bool, Dict]:
        """Transform and perform a given action in the wrapped environment. Returns
        transformed states and rewards from wrapped environment.

        Args:
            action: Action executed by the agent.

        Returns:
            observation: New state
            reward: Reward we get from performing the action
            is done: Is the simulation finished
            info: Additional diagnostic information

        Note:
            When true_reward flag is set to True it returns non-transformed reward for the testing
            purposes.
        """
        transformed_action = self.action_transformer(action, self._state_history)
        state, reward, done, info = timeit(self._env.step, "env.step")(
            transformed_action
        )
        self._state_history.update(action, reward, done, state)
        transformed_state = self.state_transformer(state, self._state_history)
        if self.true_reward:
            transformed_reward = reward
        else:
            transformed_reward = self.reward_transformer(reward, self._state_history)
        return transformed_state, transformed_reward, done, info

    def close(self):
        """Cleans up and closes the environment"""
        self._env.close()


class FrameSkipEnvironment(Environment):
    """Environment wrapper skipping frames from original environment. Action executed
    by the agent is repeated on the skipped frames.

    Args:
        env: Environment with gym like API
        environment_id: ID of the env
        state_transformer: Object of the class StateTransformer
        reward_transformer: Object of the class RewardTransformer
        action_transformer: Object of the class ActionTransformer
        n_skip_frames: Number of frames to skip on each step.
        cumulative_reward: If True, reward returned from step() method is cumulative reward from the skipped steps.
    """

    def __init__(
        self,
        env: gym.Env,
        environment_id: str = "frameskip_gym_environment_wrapper",
        state_transformer: StateTransformer = NoOpStateTransformer(),
        reward_transformer: RewardTransformer = NoOpRewardTransformer(),
        action_transformer: ActionTransformer = NoOpActionTransformer(),
        expected_episode_length: int = 512,
        n_skip_frames: int = 0,
        cumulative_reward=False,
    ):
        super().__init__(
            env,
            environment_id,
            state_transformer,
            reward_transformer,
            action_transformer,
            expected_episode_length,
        )
        self.n_skip_frames = n_skip_frames
        self.cumulative_reward = cumulative_reward

    def step(self, action: Action) -> Tuple[State, Reward, bool, Dict]:
        transformed_action = self.action_transformer(action, self._state_history)
        list_of_steps = list()

        for i in range(self.n_skip_frames + 1):
            state, reward, done, info = self._env.step(transformed_action)
            list_of_steps.append((state, reward, done, info))
            if done:
                break

        state, reward, done, info = list_of_steps[-1]
        if self.cumulative_reward:
            reward = sum([step[1] for step in list_of_steps])
        self._state_history.update(action, reward, done, state)

        transformed_state = self.state_transformer(state, self._state_history)
        if self.true_reward:
            transformed_reward = reward
        else:
            transformed_reward = self.reward_transformer(reward, self._state_history)
        return transformed_state, transformed_reward, done, info


class TimeShiftEnvironment(Environment):
    """Environment wrapper creating lag between action passed to step() method by the agent and
    action execution in the environment. First 'lag' actions are sampled from action_space.

    Args:
        env: Environment with gym like API
        environment_id: ID of the env
        state_transformer: Object of the class StateTransformer
        reward_transformer: Object of the class RewardTransformer
        action_transformer: Object of the class ActionTransformer (don't use - not implemented action transformation)

    Note:
        Class doesn't have implemented action transformation.
    """

    def __init__(
        self,
        env: gym.Env,
        environment_id: str = "timeshift_gym_environment_wrapper",
        state_transformer: StateTransformer = NoOpStateTransformer(),
        reward_transformer: RewardTransformer = NoOpRewardTransformer(),
        action_transformer: ActionTransformer = NoOpActionTransformer(),
        expected_episode_length=512,
        lag: int = 1,
    ):
        super().__init__(
            env,
            environment_id,
            state_transformer,
            reward_transformer,
            action_transformer,
            expected_episode_length,
        )
        if not isinstance(action_transformer, NoOpActionTransformer):
            raise NotImplementedError("Action transformations not implemented")
        if lag < 1:
            raise ValueError("lag must be at least 1")
        self.lag = lag
        self.fifo = None

    def step(self, action: Action) -> Tuple[State, Reward, bool, Dict]:
        transformed_action = action
        self.fifo.insert(0, transformed_action)
        action_to_exec = self.fifo.pop()
        state, reward, done, info = self._env.step(action_to_exec)
        self._state_history.update(action, reward, done, state)
        transformed_state = self.state_transformer(state, self._state_history)
        if self.true_reward:
            transformed_reward = reward
        else:
            transformed_reward = self.reward_transformer(reward, self._state_history)
        return transformed_state, transformed_reward, done, info

    def reset(self) -> State:
        self.fifo = [self._env.action_space.sample() for _ in range(self.lag)]
        return super().reset()


class TransformedSpace(Space):
    """
    Class created to handle Environments using StateTransformers as the observation space is not
    directly specified in such a system.
    """

    def __init__(self, shape=None, dtype=None, transformed_state=None):
        super().__init__(shape=shape, dtype=dtype)
        self.transformed_state = transformed_state

    def sample(self):
        """
        Return sample state. Object of this class returns always the same object. It needs to be created every sample.
        When used inside Environment with StateTransformer every call of property `observation_space` cause the
        initialization of new object, so another sample is returned.

        Returns:
            Transformed state
        """
        return self.transformed_state

    def contains(self, state: State):
        """
        This method is not available as TransformedSpace object can't estimate whether `x` is contained
        by the state representation. It is caused because TransformedSpace object infers the state properties.
        """
        raise Exception("TransformedSpace can't run method `contains`")
