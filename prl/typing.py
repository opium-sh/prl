from abc import ABC, abstractmethod
from numbers import Real
from typing import Tuple, Dict

import gym
import numpy as np
import torch

State = np.ndarray
Action = np.ndarray
Reward = Real
Space = gym.Space


class StorageABC(ABC):
    @abstractmethod
    def update(self, action, reward, done, state):
        pass

    @abstractmethod
    def new_state_update(self, state):
        pass

    @abstractmethod
    def get_states(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_last_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_rewards(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_actions(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_dones(self) -> np.ndarray:
        pass

    @abstractmethod
    def sample_batch(
        self, replay_buffor_size: int, batch_size: int, returns: bool, next_states: bool
    ) -> tuple:
        pass

    @abstractmethod
    def __getitem__(self, indicies) -> tuple:
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


MemoryABC = StorageABC


class HistoryABC(ABC):

    _index: int
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    @abstractmethod
    def update(self, action, reward, done, state):
        pass

    @abstractmethod
    def new_state_update(self, state):
        pass

    @abstractmethod
    def get_states(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_last_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_rewards(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_actions(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_dones(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_returns(self, discount_factor: float, horizon: float) -> np.ndarray:
        pass

    @abstractmethod
    def get_total_rewards(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_number_of_episodes(self) -> int:
        pass

    @abstractmethod
    def sample_batch(
        self, replay_buffor_size: int, batch_size: int, returns: bool, next_states: bool
    ) -> tuple:
        pass

    @abstractmethod
    def get_summary(self):
        pass

    @abstractmethod
    def _enlarge(self):
        pass

    @abstractmethod
    def __getitem__(self, indicies) -> tuple:
        pass

    @abstractmethod
    def __iadd__(self, other: "HistoryABC") -> "HistoryABC":
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class StateTransformerABC(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def transform(self, state: State, history: HistoryABC) -> State:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class RewardTransformerABC(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def transform(self, reward: Reward, history: HistoryABC) -> Reward:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ActionTransformerABC(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def action_space(self, original_space: Space) -> Space:
        pass

    @abstractmethod
    def transform(self, action: Action, history: HistoryABC) -> Action:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class EnvironmentABC(ABC):

    _id: str
    _state_transformer: StateTransformerABC
    _reward_transformer: RewardTransformerABC
    _action_transformer: ActionTransformerABC
    _state_history: HistoryABC
    _env: gym.Env
    _action_dtype: type
    true_reward: bool
    initial_history_length: int

    @property
    @abstractmethod
    def id(self):
        pass

    @property
    @abstractmethod
    def state_transformer(self) -> StateTransformerABC:
        pass

    @property
    @abstractmethod
    def reward_transformer(self) -> RewardTransformerABC:
        pass

    @property
    @abstractmethod
    def action_transformer(self) -> ActionTransformerABC:
        pass

    @property
    @abstractmethod
    def action_space(self) -> Space:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        pass

    @property
    @abstractmethod
    def state_history(self) -> HistoryABC:
        pass

    @abstractmethod
    def reset(self) -> State:
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[State, Reward, bool, Dict]:
        pass

    @abstractmethod
    def close(self):
        pass


class AgentABC(ABC):

    step_count: int
    iteration_count: int
    episode_count: int

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def train(
        self, env: EnvironmentABC, n_iterations: int, callback_list: list, **kwargs
    ):
        pass

    @abstractmethod
    def train_iteration(self, env: EnvironmentABC, **kwargs) -> (float, HistoryABC):
        pass

    @abstractmethod
    def pre_train_setup(self, env: EnvironmentABC, **kwargs):
        pass

    @abstractmethod
    def post_train_cleanup(self, env: EnvironmentABC, **kwargs):
        pass

    @abstractmethod
    def act(self, state: State):
        pass

    @abstractmethod
    def play_episodes(self, env, episodes: int) -> HistoryABC:
        pass

    @abstractmethod
    def play_steps(
        self, env: EnvironmentABC, n_steps: int, history: HistoryABC
    ) -> HistoryABC:
        pass

    @abstractmethod
    def test(self, env) -> HistoryABC:
        pass


class AdvantageABC(ABC):
    @abstractmethod
    def __call__(
        self,
        rewards: np.ndarray,
        baselines: np.ndarray,
        dones: np.ndarray,
        discount_factor: float,
    ) -> np.ndarray:
        pass


class FunctionApproximatorABC(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def train(self, x: np.ndarray, *loss_args) -> float:
        pass

    @abstractmethod
    def predict(self, x):
        pass


class AgentCallbackABC(ABC):

    time_logger_cid: int
    agent_logger_cid: int
    memory_logger_cid: int
    nn_logger_cid: int
    misc_logger_cid: int
    iteration_interval: int
    number_of_test_runs: int
    needs_tests: bool

    def on_iteration_end(self, agent: AgentABC) -> bool:
        pass

    def on_training_end(self, agent: AgentABC):
        pass

    def on_training_begin(self, agent: AgentABC):
        pass


class PytorchNetABC(torch.nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor):
        pass
