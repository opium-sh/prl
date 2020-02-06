import argparse

import gym
from torch import optim

from prl.agents.agents import REINFORCEAgent
from prl.environments.environments import Environment
from prl.function_approximators import PytorchFA
from prl.function_approximators.pytorch_nn import PolicyGradientLoss
from prl.function_approximators.pytorch_nn import PytorchConv
from prl.transformers.state_transformers import PongTransformer
from prl.utils import time_logger
from prl.callbacks import ValidationLogger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_iterations", help="count of train iterations", type=int, const=5, nargs="?"
)
args = parser.parse_args()

gym_env = gym.make("Pong-v0")
env = Environment(
    gym_env, expected_episode_length=1024, state_transformer=PongTransformer()
)

test_gym_env = gym.make("Pong-v0")
test_env = Environment(
    test_gym_env, expected_episode_length=1024, state_transformer=PongTransformer()
)
obs_shape = gym_env.observation_space.shape
y_size = env.action_space.n

net = PytorchConv(
    x_shape=env.observation_space.shape, hidden_sizes=[16, 16, 16], y_size=y_size
)

optimizer = optim.Adam(params=net.parameters(), lr=0.01)
loss = PolicyGradientLoss()
policy_network = PytorchFA(net=net, loss=loss, optimizer=optimizer)

agent = REINFORCEAgent(policy_network=policy_network)
callbacks = [
    ValidationLogger(on_screen=True, iteration_interval=1, number_of_test_runs=1)
]

agent.train(
    env,
    n_iterations=args.n_iterations or 5000,
    discount_factor=1.0,
    callback_list=callbacks,
    n_episodes=5,
)

print(time_logger)
