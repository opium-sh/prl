import argparse
import os

import gym
from torch import optim

from prl.agents.agents import DQNAgent
from prl.callbacks import ValidationLogger, TensorboardLogger
from prl.environments.environments import Environment
from prl.function_approximators import PytorchFA
from prl.function_approximators.pytorch_nn import DQNLoss
from prl.function_approximators.pytorch_nn import PytorchConv
from prl.transformers.state_transformers import PongTransformer
from prl.utils import time_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_iterations", help="count of train iterations", type=int, const=5, nargs="?"
)
args = parser.parse_args()

gym_env = gym.make("Pong-v0")

obs_shape = gym_env.observation_space.shape

env = Environment(
    gym_env, expected_episode_length=1024, state_transformer=PongTransformer()
)

test_gym_env = gym.make("Pong-v0")
test_env = Environment(
    test_gym_env, expected_episode_length=1024, state_transformer=PongTransformer()
)

y_size = env.action_space.n

net = PytorchConv(
    x_shape=env.observation_space.shape, hidden_sizes=[16, 16, 16], y_size=y_size
)

optimizer = optim.Adam(params=net.parameters(), lr=0.01)
loss = DQNLoss()
q_network = PytorchFA(net=net, loss=loss, optimizer=optimizer)

os.makedirs("./dqn_agents", exist_ok=True)

callbacks = [
    ValidationLogger(on_screen=True, iteration_interval=5, number_of_test_runs=1),
    TensorboardLogger(),
]

agent = DQNAgent(
    q_network=q_network,
    replay_buffer_size=50000,
    training_set_size=1024,
    epsilon_decay=50,
    start_epsilon=0.5,
    target_network_copy_iter=10,
)

agent.train(
    env,
    n_iterations=args.n_iterations or 5000,
    discount_factor=1.0,
    callback_list=callbacks,
)

print(time_logger)
