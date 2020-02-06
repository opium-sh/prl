import gym
import argparse
from torch import optim
from torch.nn import CrossEntropyLoss, Softmax

from prl.agents.agents import CrossEntropyAgent
from prl.environments.environments import TimeShiftEnvironment
from prl.function_approximators import PytorchFA
from prl.function_approximators.pytorch_nn import PytorchMLP
from prl.utils import time_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_iterations", help="count of train iterations", type=int, const=20, nargs="?"
)
args = parser.parse_args()
gym_env = gym.make("CartPole-v0")
env = TimeShiftEnvironment(gym_env, lag=5)

y_size = env.action_space.n

net = PytorchMLP(
    x_shape=env.observation_space.shape,
    hidden_sizes=[128],
    y_size=y_size,
    output_activation=Softmax(dim=1),
)

optimizer = optim.Adam(params=net.parameters(), lr=0.01)
loss = CrossEntropyLoss()
policy_network = PytorchFA(net=net, loss=loss, optimizer=optimizer)

agent = CrossEntropyAgent(policy_network=policy_network)

agent.train(env, n_iterations=args.n_iterations or 20, n_episodes=32, percentile=90)

print(time_logger)
