import gym
import numpy as np
import torch.nn as nn
import argparse
from torch import optim

from prl.agents.agents import REINFORCEAgent
from prl.callbacks import ValidationLogger, TensorboardLogger
from prl.environments.environments import Environment
from prl.function_approximators import PytorchFA
from prl.function_approximators.pytorch_nn import PolicyGradientLoss
from prl.function_approximators.pytorch_nn import PytorchMLP
from prl.transformers.reward_transformers import RewardShiftTransformer
from prl.transformers.state_transformers import StateShiftTransformer
from prl.utils import time_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_iterations", help="count of train iterations", type=int, const=100, nargs="?"
)
args = parser.parse_args()

gym_env = gym.make("CartPole-v0")

obs_shape = gym_env.observation_space.shape

state_transformer = StateShiftTransformer(np.zeros(obs_shape) - 0.1)
reward_transformer = RewardShiftTransformer(-0.5)

env = Environment(
    gym_env,
    state_transformer=state_transformer,
    reward_transformer=reward_transformer,
    expected_episode_length=128,
)

y_size = env.action_space.n

net = PytorchMLP(
    x_shape=env.observation_space.shape,
    y_size=y_size,
    output_activation=nn.Softmax(dim=1),
    hidden_sizes=[64, 64],
)

optimizer = optim.Adam(params=net.parameters(), lr=0.01)
loss = PolicyGradientLoss()
policy_network = PytorchFA(net=net, loss=loss, optimizer=optimizer)

agent = REINFORCEAgent(policy_network=policy_network)

callbacks = [ValidationLogger(), TensorboardLogger(number_of_test_runs=5)]

agent.train(
    env,
    n_iterations=args.n_iterations or 30,
    n_episodes=32,
    discount_factor=1.0,
    callback_list=callbacks,
)


print(time_logger)
