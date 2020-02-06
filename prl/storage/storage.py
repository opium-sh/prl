from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numba import njit

from prl.typing import HistoryABC, Action, Reward, State, MemoryABC, StorageABC
from prl.utils import timeit


@njit
def calculate_returns(
    all_rewards: np.ndarray,
    dones: np.ndarray,
    horizon: Union[int, np.float],
    discount_factor: float,
    _index: int,
):
    if np.any(dones) and (horizon is np.inf) and dones[_index - 1]:
        assert 0.0 <= discount_factor <= 1.0
        splits = [-1] + list(np.nonzero(dones)[0])
        all_returns = np.zeros_like(all_rewards)
        for s in range(len(splits) - 1):
            start = splits[s] + 1
            end = splits[s + 1] + 1
            rewards = all_rewards[start:end]
            returns = np.zeros_like(rewards)
            discounts = np.zeros_like(rewards) + discount_factor
            powers = np.arange(rewards.shape[0])
            discounts = np.power(discounts, powers)
            length = len(rewards)
            for i in range(length):
                trimmed_rewards = rewards[i:length]
                trimmed_discounts = discounts[: (length - i)]
                returns[i] = np.sum(trimmed_discounts * trimmed_rewards)
            all_returns[start:end] = returns
        return all_returns
    else:
        raise Exception(
            "Returns available only for at least one complete episode, there can't be an incomplete episode"
            "and the horizon must be np.inf"
        )


@njit
def calculate_total_rewards(all_rewards: np.ndarray, dones: np.ndarray, _index: int):
    if np.any(dones) and dones[_index - 1]:
        splits = [-1] + list(np.nonzero(dones)[0])
        all_total_rewards = np.zeros_like(all_rewards)
        for s in range(len(splits) - 1):
            start = splits[s] + 1
            end = splits[s + 1] + 1
            rewards = all_rewards[start:end]
            total_reward = np.sum(rewards)
            all_total_rewards[start:end] = total_reward
        return all_total_rewards
    else:
        raise Exception(
            "Returns available only for at least one complete episode and all episodes must be done"
        )


class Storage(StorageABC, ABC):
    @abstractmethod
    def update(self, action, reward, done, state):
        """
        Updates the object with latest states, reward, actions and done flag.

        Args:
            action: action executed by the agent
            reward: reward from environments
            done: done flag from environments
            state: new state returned by wrapped environments after executing action
        """

    @abstractmethod
    def new_state_update(self, state):
        """Overwrites newest state in the History

        Args:
            state: state array.
        """

    @abstractmethod
    def get_states(self) -> np.ndarray:
        """Returns an array of all states.

        Returns:
            array of all states
        """

    @abstractmethod
    def get_last_state(self) -> np.ndarray:
        """Returns only the last state.

        Returns:
            last state
        """

    @abstractmethod
    def get_rewards(self) -> np.ndarray:
        """Returns an array of all rewards.

        Returns:
            array of all rewards
        """

    @abstractmethod
    def get_actions(self) -> np.ndarray:
        """Returns an array of all actions.

        Returns:
            array of all actions
        """

    @abstractmethod
    def get_dones(self) -> np.ndarray:
        """Returns an array of all done flags.

        Returns:
            array of all done flags
        """

    @abstractmethod
    def sample_batch(
        self, replay_buffor_size: int, batch_size: int, returns: bool, next_states: bool
    ) -> tuple:
        """Samples batch of examples from the Storage.

        Args:
            replay_buffer_size: length of a replay buffor to sample examples from
            batch_size: number of returned examples
            returns: if True, the method will return the returns from each step instead of the rewards
            next_states: if True, the method will return also next states (i.e. for DQN algorithm)

        Returns:
            batch of samples from history in form of a tuple with np.ndarrays in order:
            states, actions, rewards, dones, (new_states)
        """

    @timeit
    def __getitem__(self, indicies) -> tuple:
        return (
            self.get_states()[indicies],
            self.get_actions()[indicies],
            self.get_rewards()[indicies],
            self.get_dones()[indicies],
        )

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class History(Storage, HistoryABC):
    """
    An object which is used to keep the episodes history (used within :py:class:`~prl.environments.environments.Environment` class
    and by some agents). Agent can use this object to keep history of past episodes,
    calculate returns, total rewards, etc. and sample batches from it.

    Object also supports indexing and slicing because it supports python Sequence protocol,
    so functions working on sequences like random.choice can be also used on history.

    Args:
        initial_state: initial state from enviroment
        action_type: numpy type of action (e.g. np.int32)
        initial_length: initial length of a history
    """

    @timeit
    def __init__(
        self, initial_state: np.ndarray, action_type: type, initial_length: int = 512
    ):
        self._index = 0

        self.states = np.empty(
            (initial_length,) + initial_state.shape, dtype=np.float32
        )
        self.actions = np.empty((initial_length,), dtype=action_type)
        self.rewards = np.empty((initial_length,))
        self.dones = np.empty((initial_length,), dtype=np.bool)
        self.states[self._index] = initial_state

    @timeit
    def update(self, action: Action, reward: Reward, done: bool, state: State):
        if self._index == (self.states.shape[0] - 1):
            self._enlarge()
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.dones[self._index] = done
        self._index += 1
        self.states[self._index] = state

    @timeit
    def new_state_update(self, state: State):
        self.states[self._index] = state

    @timeit
    def get_states(self) -> np.ndarray:
        return self.states[: self._index]

    @timeit
    def get_last_state(self) -> np.ndarray:
        return self.states[self._index]

    @timeit
    def get_rewards(self) -> np.ndarray:
        return self.rewards[: self._index]

    @timeit
    def get_actions(self) -> np.ndarray:
        return self.actions[: self._index]

    @timeit
    def get_dones(self) -> np.ndarray:
        return self.dones[: self._index]

    @timeit
    def get_returns(
        self, discount_factor: float = 1.0, horizon: float = np.inf
    ) -> np.ndarray:
        """Calculates returns for each step.

        Returns:
            array of discounted returns for each step
        """
        return calculate_returns(
            self.get_rewards(), self.get_dones(), horizon, discount_factor, self._index
        )

    @timeit
    def get_total_rewards(self) -> np.ndarray:
        """
        Calculates sum of all rewards for each episode and reports it for each state,
        so every state in one episode has the same value of total reward. This can
        be useful for filtering states for best episodes (e.g. in Cross Entropy Algorithm).

        Returns:
            total reward for each state
        """
        return calculate_total_rewards(
            self.get_rewards(), self.get_dones(), self._index
        )

    @timeit
    def get_number_of_episodes(self) -> int:
        """Returns a number of full episodes in history.

        Returns:
            number of full episodes in history
        """
        return int(self.get_dones().sum())

    @timeit
    def sample_batch(
        self,
        replay_buffer_size: int,
        batch_size: int = 64,
        returns: bool = False,
        next_states: bool = False,
    ) -> tuple:
        if returns:
            raise NotImplementedError("The returns will be implemented soon")
        elif next_states:
            if self._index < 2:
                raise Exception(
                    "Can't sample examples with next_state when the history has length 1."
                )
            indexes = np.random.randint(
                np.max([0, self._index - replay_buffer_size]),
                self._index - 1,
                size=batch_size,
            )
            return self[indexes] + (self.get_states()[indexes + 1],)
        else:
            indexes = np.random.randint(
                np.max(0, self._index - replay_buffer_size),
                self._index,
                size=batch_size,
            )
            return self[indexes]

    def get_summary(self) -> (float, float, int):
        total_rewards_mean = self.get_total_rewards()[self.get_dones()].mean()
        mean_length = len(self) / self.get_number_of_episodes()
        return total_rewards_mean, mean_length, self._index

    @timeit
    def _enlarge(self):
        new_shape = list(self.states.shape)
        new_shape[0] *= 2
        self.states = np.resize(self.states, new_shape)
        new_shape = list(self.actions.shape)
        new_shape[0] *= 2
        self.actions = np.resize(self.actions, new_shape)
        new_shape = list(self.rewards.shape)
        new_shape[0] *= 2
        self.rewards = np.resize(self.rewards, new_shape)
        new_shape = list(self.dones.shape)
        new_shape[0] *= 2
        self.dones = np.resize(self.dones, new_shape)
        print("Enlarging History. New max length: ", self.dones.shape[0])

    def __add__(self, other):
        raise NotImplementedError(
            "You can only use inplace operators between History instances"
        )

    @timeit
    def __iadd__(self, other: HistoryABC):
        self.states = np.concatenate([self.get_states(), other.states])
        self.actions = np.concatenate([self.get_actions(), other.actions])
        self.rewards = np.concatenate([self.get_rewards(), other.rewards])
        self.dones = np.concatenate([self.get_dones(), other.dones])
        self._index += other._index
        return self

    def __len__(self):
        return self._index

    def __repr__(self):
        representation = ""
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                representation += "%s:\n%s\n" % (k, v[: self._index])
            else:
                representation += "%s:\n%s\n" % (k, v)
        return representation


class Memory(Storage, MemoryABC):
    """
    An object to be used as replay buffer. Doesn't contain full episodes and acts
    as limited FIFO queue. Implemented as double size numpy arrays with duplicated data
    to support very fast slicing and sampling at the cost of higher memory usage.

    Args:
        initial_state: initial state from enviroment
        action_type: numpy type of action (e.g. np.int32)
        maximum_length: maximum number of examples to keep in queue
    """

    @timeit
    def __init__(
        self, initial_state: np.ndarray, action_type, maximum_length: int = 1000
    ):
        self._maximum_length = maximum_length

        self.states = np.empty(
            (2 * maximum_length + 2,) + initial_state.shape, dtype=np.float32
        )
        self.actions = np.empty((2 * maximum_length + 2,), dtype=action_type)
        self.rewards = np.empty((2 * maximum_length + 2,))
        self.dones = np.empty((2 * maximum_length + 2,), dtype=np.bool)

        self.clear(initial_state)

    @timeit
    def clear(self, initial_state):
        self._lower_index = 0
        self._index = 1
        self._full = False
        self.states[self._index] = initial_state

    @timeit
    def update(self, action, reward, done, state):
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.dones[self._index] = done
        if self._full:
            self.actions[self._lower_index] = action
            self.rewards[self._lower_index] = reward
            self.dones[self._lower_index] = done
        self._index += 1
        if self._index > self._maximum_length + 1:
            self._lower_index += 1
            self._full = True
            self.states[self._lower_index] = state
        if self._index == 2 * self._maximum_length + 2:
            self._index = self._maximum_length + 1
            self._lower_index = 0
        self.states[self._index] = state

    @timeit
    def new_state_update(self, state):
        self.states[self._index] = state
        if self._full:
            self.states[self._lower_index] = state

    @timeit
    def get_states(self, include_last=False) -> np.ndarray:
        index = self._index
        if include_last:
            index += 1
        return self.states[(self._lower_index + 1) : index]

    @timeit
    def get_last_state(self) -> np.ndarray:
        return self.states[self._index]

    @timeit
    def get_rewards(self) -> np.ndarray:
        return self.rewards[(self._lower_index + 1) : self._index]

    @timeit
    def get_actions(self) -> np.ndarray:
        return self.actions[(self._lower_index + 1) : self._index]

    @timeit
    def get_dones(self) -> np.ndarray:
        return self.dones[(self._lower_index + 1) : self._index]

    @timeit
    def sample_batch(
        self,
        replay_buffor_size: int,
        batch_size: int = 64,
        returns: bool = False,
        next_states: bool = False,
    ) -> tuple:
        if returns:
            raise NotImplementedError("The returns will be implemented soon")
        elif next_states:
            if self._index < 2:
                raise Exception(
                    "Can't sample examples with next_state when the history has length 1."
                )
            indicies = np.random.randint(
                self._index - self._lower_index - 2, size=batch_size
            )
            return self[indicies] + (self.get_states()[indicies + 1],)
        else:
            indicies = np.random.randint(
                self._index - self._lower_index, size=batch_size
            )
            return self[indicies]

    def __len__(self):
        return self._index - (self._lower_index + 1)

    def __repr__(self):
        representation = ""
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                representation += "%s:\n%s\n" % (
                    k,
                    v[(self._lower_index + 1) : self._index],
                )
            else:
                representation += "%s:\n%s\n" % (k, v)
        return representation
