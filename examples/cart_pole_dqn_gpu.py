import gym
import torch
import argparse
from torch import optim

from prl.agents.agents import DQNAgent
from prl.environments.environments import Environment
from prl.function_approximators import PytorchFA
from prl.function_approximators.pytorch_nn import DQNLoss
from prl.function_approximators.pytorch_nn import PytorchMLP
from prl.utils import time_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_iterations", help="count of train iterations", type=int, const=3000, nargs="?"
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

q_network = PytorchFA(net=net, loss=loss, optimizer=optimizer, device=device)

agent = DQNAgent(
    q_network=q_network,
    replay_buffer_size=5000,
    training_set_size=256,
    epsilon_decay=2000,
    target_network_copy_iter=10,
)

agent.train(env, n_iterations=args.n_iterations or 3000, discount_factor=1.0)

print(time_logger)
