from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce
from operator import iadd

import numpy as np

from prl.callbacks.callbacks import CallbackHandler
from prl.environments.environments import Environment
from prl.storage import History, Memory, Storage
from prl.typing import (
    EnvironmentABC,
    FunctionApproximatorABC,
    AgentABC,
    AdvantageABC,
    State,
    Action,
)
from prl.utils import timeit, agent_logger


class Agent(AgentABC, ABC):
    """Base class for all agents"""

    def __init__(self):
        self.step_count = 0
        self.iteration_count = 0
        self.episode_count = 0
        self._actual_reward_count = 0
        self._actual_episode_length = 0

    @property
    @abstractmethod
    def id(self) -> str:
        """Agent UUID"""

    @timeit
    def train(
        self, env: Environment, n_iterations: int, callback_list: list = None, **kwargs
    ):
        """Trains the agent using environment. Also handles callbacks during training.

        Args:
            env: Environment to train on
            n_iterations: Maximum number of iterations to train
            callback_list: List of callbacks
            kwargs: other arguments passed to `train_iteration`, `pre_train_setup` and `post_train_cleanup`
        """
        agent_logger.add("agent_step", self.step_count)
        agent_logger.add("agent_iteration", self.iteration_count)
        agent_logger.add("agent_episode", self.episode_count)

        callback_list = callback_list or []
        callback_handler = CallbackHandler(callback_list, env)

        self.pre_train_setup(env, **kwargs)

        callback_handler.on_training_begin(self)

        for i in range(n_iterations):
            self.train_iteration(env=env, **kwargs)
            self.iteration_count += 1
            agent_logger.add("agent_iteration", self.iteration_count)
            if callback_handler.on_iteration_end(self):
                break

        self.post_train_cleanup(env, **kwargs)
        callback_handler.on_training_end(self)

    @abstractmethod
    def train_iteration(self, env: Environment, **kwargs):
        """Performs single training iteration. This method should contain repeatable
        part of training an agent.

        Args:
            env: Environment
            **kwargs: Kwargs passed from train() method
        """

    def pre_train_setup(self, env: Environment, **kwargs):
        """Performs pre-training setup. This method should handle non-repeatable part of
        training an agent.

        Args:
            env: Environment
            **kwargs: Kwargs passed from train() method
        """

    def post_train_cleanup(self, env: Environment, **kwargs):
        """Performs cleaning up fields that are no longer needed after training to keep
        agent lightweight.

        Args:
            env: Environment
            **kwargs: Kwargs passed from train() method
        """

    @abstractmethod
    def act(self, state: State) -> Action:
        """Makes a step based on current environments state

        Args:
            state: state from the environment.

        Returns:
            Action to execute on the environment.
        """

    @timeit
    def play_episodes(self, env: Environment, episodes: int) -> History:
        """Method for playing full episodes used usually to train agents.

        Args:
            env: Environment
            episodes: Number of episodes to play.
        Returns:
            History object representing episodes history
        """
        history_list = []
        for i in range(episodes):
            state = env.reset()
            history: History = History(state, np.int32, env.initial_history_length)
            while True:
                action = self.act(state)
                state, reward, done, _ = env.step(action)
                self.step_count += 1
                agent_logger.add("agent_step", self.step_count)
                history.update(action, reward, done, state)
                # TODO step callback
                if done:
                    self.episode_count += 1
                    agent_logger.add("agent_episode", self.episode_count)
                    history_list.append(history)
                    agent_logger.add(
                        "episode_total_reward", history.get_total_rewards()[-1]
                    )
                    agent_logger.add("episode_length", len(history))
                    # TODO episode callback
                    break
        return reduce(iadd, history_list)

    @timeit
    def play_steps(self, env: Environment, n_steps: int, storage: Storage) -> Storage:
        """Method for performing some number of steps in the environments. Appends new
        states to existing storage
        Args:
            env: Environment
            n_steps: Number of steps to play
            storage: Storage (Memory, History) of the earlier games (used to perform first action)
        Returns:
            History with appended states, actions, rewards, etc
        """
        state = storage.get_last_state()
        for i in range(n_steps):
            action = self.act(state)
            state, reward, done, _ = env.step(action)
            self._actual_reward_count += reward
            self._actual_episode_length += 1
            self.step_count += 1
            agent_logger.add("agent_step", self.step_count)
            storage.update(action, reward, done, state)
            # TODO step callback
            if done:
                self.episode_count += 1
                agent_logger.add("agent_episode", self.episode_count)
                # TODO episode callback
                agent_logger.add("episode_total_reward", self._actual_reward_count)
                agent_logger.add("episode_length", self._actual_episode_length)
                state = env.reset()
                self._actual_reward_count = 0
                self._actual_episode_length = 0
                storage.new_state_update(state)
        return storage

    @timeit
    def test(self, env) -> History:
        """
        Method for playing full episode used to test agents. Reward in the returned history is
        the true reward from the environments. This method is used mostly for testing the agent.

        Args:
            env: Environment
        Returns:
            History object representing episode history
        """
        env.true_reward = True
        history = self.play_episodes(env, 1)
        agent_logger.add("test_episode_total_reward", history.get_total_rewards()[-1])
        agent_logger.add("test_episode_length", len(history))
        env.true_reward = False
        return history


class RandomAgent(Agent):
    """Agent performing random actions"""

    def __init__(self, agent_id: str = "random_agent", replay_buffer_size=100):
        super().__init__()
        self._id = agent_id
        self.action_space = None
        self.replay_buffer_size = replay_buffer_size

    @property
    def id(self):
        return self._id

    @timeit
    def pre_train_setup(self, env: Environment, **kwargs):
        self.action_space = env.action_space
        state = env.reset()
        self.replay_buffer = Memory(state, np.int32, self.replay_buffer_size)
        # To ensure that we have the next state after doing the first step.
        self.play_steps(env, n_steps=1, storage=self.replay_buffer)

    @timeit
    def train_iteration(self, env: Environment, discount_factor: float = 1.0):
        self.play_steps(env, 1, self.replay_buffer)
        return None, self.replay_buffer

    def act(self, state: State):
        return self.action_space.sample()


class CrossEntropyAgent(Agent):
    """Agent using cross entropy algorithm"""

    def __init__(
        self,
        policy_network: FunctionApproximatorABC,
        agent_id: str = "crossentropy_agent",
    ):
        super().__init__()
        self._id = agent_id
        self.action_space = None
        self.policy_network = policy_network

    @property
    def id(self):
        return self._id

    @timeit
    def train_iteration(self, env: EnvironmentABC, n_episodes=32, percentile=75):
        history = self.play_episodes(env, n_episodes)
        all_total_rewards = history.get_total_rewards()
        total_rewards = all_total_rewards[history.get_dones()]
        total_reward_bound = np.percentile(total_rewards, percentile)
        above_treshold_mask = all_total_rewards >= total_reward_bound
        states = history.get_states()[above_treshold_mask]
        actions = history.get_actions()[above_treshold_mask]
        loss = self.policy_network.train(states, actions)
        return loss, history

    def act(self, state: State) -> Action:
        state = state.reshape(1, *state.shape)
        act_probs = self.policy_network.predict(state)[0]
        return np.random.choice(len(act_probs), p=act_probs)


class REINFORCEAgent(Agent):
    """Agent using REINFORCE algorithm"""

    def __init__(
        self, policy_network: FunctionApproximatorABC, agent_id: str = "REINFORCE_agent"
    ):
        super().__init__()
        self._id = agent_id
        self.action_space = None
        self.policy_network = policy_network

    @property
    def id(self):
        return self._id

    def pre_train_setup(
        self, env: EnvironmentABC, discount_factor: float = 1.0, **kwargs
    ):
        assert 0.0 <= discount_factor <= 1.0

    @timeit
    def train_iteration(
        self, env: EnvironmentABC, n_episodes: int = 32, discount_factor: float = 1.0
    ):
        history = self.play_episodes(env, n_episodes)
        states = history.get_states()
        actions = history.get_actions()
        returns = history.get_returns(discount_factor)
        loss = self.policy_network.train(states, actions, returns)
        return loss, history

    @timeit
    def act(self, state: State) -> Action:
        state = state.reshape(1, *state.shape)
        act_probs = self.policy_network.predict(state)[0]
        return np.random.choice(len(act_probs), p=act_probs)


class ActorCriticAgent(Agent):
    """Basic actor-critic agent."""

    def __init__(
        self,
        policy_network: FunctionApproximatorABC,
        value_network: FunctionApproximatorABC,
        advantage: AdvantageABC,
        agent_id: str = "ActorCritic_agent",
    ):
        super().__init__()
        self._id = agent_id
        self.policy_network = policy_network
        self.value_network = value_network
        self.advantage = advantage
        self.memory = None
        self._should_reset = True

    @property
    def id(self):
        return self._id

    @timeit
    def train_iteration(
        self, env: EnvironmentABC, n_steps: int = 32, discount_factor: float = 1.0
    ):
        if self._should_reset:
            self.memory = Memory(
                initial_state=env.reset(), action_type=np.int32, maximum_length=n_steps
            )
            self._should_reset = False
        self.play_steps(env, n_steps, self.memory)
        states = self.memory.get_states(include_last=True)
        values = self.value_network.predict(states).squeeze(axis=-1)
        states = states[:-1, ...]
        actions = self.memory.get_actions()
        rewards = self.memory.get_rewards()
        dones = self.memory.get_dones()
        advantages = self.advantage(rewards, values, dones, discount_factor)
        values = values[:-1]
        policy_loss = self.policy_network.train(states, actions, advantages)
        target_values = values + advantages
        value_loss = self.value_network.train(states, target_values)
        # TODO: entropy loss (add to PolicyGradientLoss)
        return (None, self.memory)

    @timeit
    def act(self, state: State) -> Action:
        state = state.reshape(1, *state.shape)
        act_probs = self.policy_network.predict(state)[0]
        return np.random.choice(len(act_probs), p=act_probs)


class A2CAgent(ActorCriticAgent):
    """Advantage Actor Critic agent."""

    def __init__(
        self,
        policy_network: FunctionApproximatorABC,
        value_network: FunctionApproximatorABC,
        agent_id: str = "A2C_agent",
    ):
        super().__init__(
            policy_network, value_network, advantage=A2CAdvantage(), agent_id=agent_id
        )


class Advantage(AdvantageABC, ABC):
    """Base class for advantage functions."""

    def __call__(
        self,
        rewards: np.ndarray,
        baselines: np.ndarray,
        dones: np.ndarray,
        discount_factor: float,
    ) -> np.ndarray:
        assert (
            rewards.shape[:-1] == baselines.shape[:-1]
        ), "Incompatible shapes of rewards and baselines."
        assert (
            baselines.shape[-1] == rewards.shape[-1] + 1
        ), "Baseline sequence should be 1 longer than the reward sequence."
        assert rewards.shape == dones.shape, "Incompatible shapes of rewards and dones."
        return self.calculate_advantages(rewards, baselines, dones, discount_factor)

    @abstractmethod
    def calculate_advantages(
        self,
        rewards: np.ndarray,
        baselines: np.ndarray,
        dones: np.ndarray,
        discount_factor: float,
    ) -> np.ndarray:
        pass


class A2CAdvantage(Advantage):
    """Advantage function from Asynchronous Methods for Deep Reinforcement Learning."""

    @timeit
    def calculate_advantages(
        self,
        rewards: np.ndarray,
        baselines: np.ndarray,
        dones: np.ndarray,
        discount_factor: float,
    ) -> np.ndarray:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        current_return = baselines[-1]
        for i in reversed(range(len(rewards))):
            current_return = rewards[i] + ~dones[i] * discount_factor * current_return
            advantages[i] = current_return - baselines[i]
        return advantages


class GAEAdvantage(Advantage):
    """Advantage function from High-Dimensional Continuous Control Using
    Generalized Advantage Estimation.
    """

    def __init__(self, lambda_: float):
        self.lambda_ = lambda_

    @timeit
    def calculate_advantages(
        self,
        rewards: np.ndarray,
        baselines: np.ndarray,
        dones: np.ndarray,
        discount_factor: float,
    ) -> np.ndarray:
        deltas = rewards + ~dones * discount_factor * baselines[1:] - baselines[:-1]
        gamma_lambda = discount_factor * self.lambda_
        advantages = np.zeros_like(rewards, dtype=np.float32)
        current_advantage = 0
        for i in reversed(range(len(rewards))):
            current_advantage = deltas[i] + ~dones[i] * gamma_lambda * current_advantage
            advantages[i] = current_advantage
        return advantages


class DQNAgent(Agent):
    """Agent using DQN algorithm"""

    def __init__(
        self,
        q_network: FunctionApproximatorABC,
        replay_buffer_size: int = 10000,
        start_epsilon: float = 1.0,
        end_epsilon: float = 0.05,
        epsilon_decay: int = 1000,
        training_set_size: int = 64,
        target_network_copy_iter: int = 100,
        steps_between_training=10,
        agent_id: str = "DQN_agent",
    ):
        super().__init__()
        self._id = agent_id
        self.action_space = None
        self.start_epsilon = start_epsilon
        self.epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay
        self.replay_buffer_size = replay_buffer_size
        self.q_network = q_network
        self.target_network = deepcopy(q_network)
        self.batch_size = training_set_size
        self.target_network_copy_iter = target_network_copy_iter
        self.steps_between_training = steps_between_training

        self.epsilon_diff = (self.start_epsilon - self.end_epsilon) / self.epsilon_decay
        self.replay_buffer = None

    @property
    def id(self):
        return self._id

    def pre_train_setup(
        self, env: EnvironmentABC, discount_factor: float = 1.0, **kwargs
    ):
        assert 0.0 <= discount_factor <= 1.0
        state = env.reset()
        self.replay_buffer = Memory(state, np.int32, self.replay_buffer_size)
        # To ensure that we have the next state after doing the first step.
        self.play_steps(env, n_steps=1, storage=self.replay_buffer)

    @timeit
    def train_iteration(self, env: EnvironmentABC, discount_factor: float = 1.0):

        if self.epsilon_decay < self.iteration_count:
            self.epsilon -= self.epsilon_diff
        if self.iteration_count % self.target_network_copy_iter == 0:
            self.target_network = deepcopy(self.q_network)
        self.play_steps(env, self.steps_between_training, self.replay_buffer)
        states, actions, rewards, dones, next_states = self.replay_buffer.sample_batch(
            self.replay_buffer_size, self.batch_size, next_states=True
        )
        target_vals = self.target_network.predict(next_states)
        target_ind = np.argmax(target_vals, axis=1)
        target_max = target_vals[np.arange(target_vals.shape[0]), target_ind]
        target_q = rewards + discount_factor * target_max * (~dones)
        loss = self.q_network.train(states, actions, target_q)
        return loss, self.replay_buffer

    def act(self, state: State) -> Action:
        state = state.reshape(1, *state.shape)
        act_qvals = self.q_network.predict(state)[0]
        if np.random.uniform() < self.epsilon:
            return np.random.choice(len(act_qvals))
        else:
            return np.argmax(act_qvals)
