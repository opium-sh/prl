import argparse
import os

import gym
import numpy as np
from torch import nn
from torch import optim

from prl.agents import CrossEntropyAgent
from prl.callbacks import (
    TrainingLogger,
    EarlyStopping,
    PyTorchAgentCheckpoint,
    TensorboardLogger,
    ValidationLogger,
)
from prl.environments import Environment
from prl.function_approximators import PytorchFA
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

state_transformer = StateShiftTransformer(np.zeros(obs_shape) - 0.5)
reward_transformer = RewardShiftTransformer(-0.5)

env = Environment(
    gym_env, state_transformer=state_transformer, reward_transformer=reward_transformer
)

y_size = env.action_space.n

net = PytorchMLP(
    x_shape=env.observation_space.shape,
    hidden_sizes=[128],
    output_activation=nn.Softmax(dim=1),
    y_size=y_size,
)

optimizer = optim.Adam(params=net.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()
policy_network = PytorchFA(net=net, loss=loss, optimizer=optimizer)

agent = CrossEntropyAgent(policy_network=policy_network)

os.makedirs("./test_pytorch_saving", exist_ok=True)

callbacks = [
    TrainingLogger(
        on_screen=True, iteration_interval=1, to_file=True, file_path="./test_logs.txt"
    ),
    ValidationLogger(on_screen=True, iteration_interval=5),
    EarlyStopping(target_reward=150, iteration_interval=5, number_of_test_runs=5),
    PyTorchAgentCheckpoint(
        target_path="./test_pytorch_saving",
        save_best_only=False,
        iteration_interval=10,
        number_of_test_runs=5,
    ),
    TensorboardLogger(iteration_interval=5),
]

agent.train(
    env,
    n_iterations=args.n_iterations or 100,
    n_episodes=32,
    percentile=90,
    callback_list=callbacks,
)

print(time_logger)

time_logger.save("./time_logger.pkl")
