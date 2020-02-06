import argparse

import gym
import torch
from torch import nn
from torch import optim

from prl.agents import A2CAgent
from prl.callbacks import TrainingLogger, ValidationLogger
from prl.environments import Environment
from prl.function_approximators import PytorchFA
from prl.function_approximators.pytorch_nn import PolicyGradientLoss
from prl.function_approximators.pytorch_nn import PytorchMLP
from prl.utils import time_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_iterations", help="count of train iterations", type=int, const=5000, nargs="?"
)
args = parser.parse_args()

gym_env = gym.make("CartPole-v0")
env = Environment(gym_env, expected_episode_length=256)

y_size = env.action_space.n

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

policy_net = PytorchMLP(
    x_shape=env.observation_space.shape,
    y_size=y_size,
    output_activation=nn.functional.softmax,
    hidden_sizes=[24],
)
value_net = PytorchMLP(
    x_shape=env.observation_space.shape,
    y_size=1,
    output_activation=lambda x: x,
    hidden_sizes=[24],
)
policy_net = PytorchFA(
    net=policy_net,
    loss=PolicyGradientLoss(),
    optimizer=optim.Adam(params=policy_net.parameters(), lr=0.001),
    device=device,
)
value_net = PytorchFA(
    net=value_net,
    loss=nn.MSELoss(),
    optimizer=optim.Adam(params=value_net.parameters(), lr=0.005),
    device=device,
)


agent = A2CAgent(policy_network=policy_net, value_network=value_net)

callbacks = [TrainingLogger(), ValidationLogger(iteration_interval=500)]

agent.train(
    env,
    n_iterations=args.n_iterations or 10000,
    discount_factor=0.99,
    callback_list=callbacks,
    n_steps=16,
)

print(time_logger)
