import argparse

import gym

from prl.agents import RandomAgent
from prl.environments.environments import Environment
from prl.transformers.state_transformers import PongTransformer
from prl.utils import time_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_iterations", help="count of train iterations", type=int, const=100, nargs="?"
)
args = parser.parse_args()

gym_env = gym.make("Pong-v0")
env = Environment(
    gym_env, expected_episode_length=1024, state_transformer=PongTransformer()
)

agent = RandomAgent()
agent.train(env, n_iterations=args.n_iterations or 100)

print(time_logger)
