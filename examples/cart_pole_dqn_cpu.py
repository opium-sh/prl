import argparse

import gym
from torch import optim

from prl.agents import DQNAgent
from prl.callbacks import ValidationLogger
from prl.environments import Environment
from prl.function_approximators import PytorchFA
from prl.function_approximators.pytorch_nn import DQNLoss
from prl.function_approximators.pytorch_nn import PytorchMLP
from prl.utils import time_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_iterations", help="count of train iterations", type=int, const=5000, nargs="?"
)
args = parser.parse_args()

gym_env = gym.make("CartPole-v0")
env = Environment(gym_env, expected_episode_length=128)

y_size = env.action_space.n

net = PytorchMLP(
    x_shape=env.observation_space.shape,
    y_size=y_size,
    output_activation=lambda x: x,
    hidden_sizes=[64],
)

optimizer = optim.Adam(params=net.parameters(), lr=0.01)
loss = DQNLoss(mode="mse")  # MSE works better than Huber loss on CartPole.
q_network = PytorchFA(net=net, loss=loss, optimizer=optimizer, batch_size=256)

agent = DQNAgent(
    q_network=q_network,
    replay_buffer_size=1000,
    training_set_size=256,
    epsilon_decay=1000,
    target_network_copy_iter=500,
)

callbacks = [ValidationLogger(iteration_interval=500)]

agent.train(
    env,
    n_iterations=args.n_iterations or 10000,
    discount_factor=1.0,
    callback_list=callbacks,
)

print(time_logger)
