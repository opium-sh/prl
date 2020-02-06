# People's Reinforcement Learning (PRL)

![](https://img.shields.io/badge/python-3.6-blue.svg)
![](https://img.shields.io/badge/code%20style-black-000000.svg)
![](https://readthedocs.org/projects/prl/badge/?version=latest)

## Description

Our main goal is to build a useful tool for the reinforcement learning researchers.

While using PRL library for building agents and conducting experiments you
can focus on a structure of an agent, state transformations, neural networks
architecture, action transformations and reward shaping. Time and memory profiling, 
logging, agent-environment interactions, agent state saving, 
neural network training, early stopping or training visualization 
happens automatically behind the scenes. You are also provided 
with very useful tools for handling training history and preparing training sets for
neural networks.

## System requirements

* ```python 3.6```
* ```swig```

We recommend using ```virtualenv``` for installing project dependencies.

## Installation

* clone the project:

  ```
  git clone git@gitlab.com:opium-sh/prl.git
  ```

* create and activate a virtualenv for the project (you can skip this step if you are not using virtualenv)

  ```
  virtualenv -p python3.6 your/path && source your/path/bin/activate
  ```

* install dependencies:

  ```
  pip install -r requirements.txt
  ```
  
* install library

  ```
  pip install -e .
  ```

* run example:

   ```
   cd examples
   python cart_pole_example_cross_entropy.py
   ```
   
# Overall structure

Two main classes in PRL are `Agent` and `Environment` class.

One of the main goals while building PRL framework was to make Agent implementations 
as clear and compact as they can be. All the state, reward and action transformations are 
within `Environment` class, which wraps inside Gym-like environment.
`Agent` and `Environment` class can use `Storage` objects to easily manage training history
in order to transform states and create training sets for neural networks.
Neural networks are also wrapped within `FunctionApproximator` class, which provides
the user with unified API. All the networks are implemented in PyTorch and all the
data transformations outside the networks are written using NumPy and Numba.

Neural networks packed within `FunctionApproximators` class can be easily transferred
between different agents, so you can pretrain a policy using one algorithm 
(using imitation learning for example) and then
use it in another agent implementing another training algorithm (e.g. actor-critic).

## Environment

`prl.environment`

`Environment` class is the base class for the PRL environments. It preserves OpenAi Gym
API with some minor exceptions. `Environment` wraps the OpenAi Gym environment inside 
and have placeholders for state, action and reward transformer objects. 
It also tracks the history of currently played episode. Classes inheriting from 
`Environment` are `GymEnvironment`, `TimeShiftEnvironment` and `FrameSkipEnvironment`.

Transformers can use the history of an episode to transform current observation
from Gym env to state representation required by agent, so transformer objects 
can be implemented as stateless functions if it is convenient for the user.
States, actions and rewards stored in the episode history are in the raw form
(before any transformations).

For now, only environments with discrete action spaces and observations being
`np.ndarray` are supported.

### State transformers

`prl.transformers.state_transformers`

State transformers are used by the environment to change observations to state 
representation for function approximators used by the agent. To build your own
transformer all you need is to create a class inheriting from `prl.core.StateTransformer` 
class and implement `transform` and `reset` methods. Transform method takes two arguments: state
in a form of `np.ndarray` and episode history. It returns a state represented also
by `np.ndarray`. `reset` method resets transformer state (if any) between episodes.
You can make your transformation fittable to the data by implementing `fit`
method in the same manner as in the scikit-learn library. 

It is important to use only NumPy functions while implementing transformers for the
sake of performance. More advanced Python programmers can use 
[Numba](http://numba.pydata.org/) library to speed
up their transformers even more. Very good and deep NumPy tutorial can be found 
[here](https://www.labri.fr/perso/nrougier/from-python-to-numpy/).

You can check the performance of your transformations by importing and using `print` 
function on `prl.utils.time_logger` object at the end of your program.

### Reward transformers

`prl.transformers.reward_transformers`

If you want to make your own reward shaping transformer you need to inherit
from `prl.core.RewardTransformer` and perform analogous steps as in the state
transformer case.

### Action transformers

`prl.transformers.action_transformers`

While implementing the action transformer you need to inherit your class from 
`prl.core.ActionTransformer`, implement `reset`, `transform` and `fit` method.
After that you have to assign a gym.Space object to the `action_space` attribute,
because it cannot be automatically inferred only from the class implementation.

## Storage

Classes for storage are created for easy management of the training history.

### History

`prl.core.History`

`History` class is used to keep the episodes history. You can get actions,
states, rewards and done flag from it. It also gives user methods to prepare array
with returns, count total rewards or sample a batch for 
neural network training. You can concatenate two history objects by using inplace add
operator `+=`.

Because appending to an `np.ndarray` (used to store data) is a very expensive operation, the `History` object
allocates in advance some bigger arrays, and doubles it's size when arrays
are full. You can set the initial length of a `History` object during 
initialization (e.g. `Environment` does it based on `expected_episode_length` parameter).

### Memory

`prl.core.Memory`

Class similar to `History` created to be used as a replay buffer. It does not have to
keep complete episodes, so it's API has less methods than `History`. It's length is
constant and it is set during object initialization. You can't concatenate two `Memory`
objects or calculate total rewards.

## Function approximators

`prl.function_approximators`

Function approximators (FA) are created to deliver unified API for any kind of function approximators
used by RL algorithms. FA have two methods: `train` and `predict` and are implemented in
[PyTorch](https://pytorch.org/) for now.

### PytorchNN

`prl.function_approximators.PytorchNN`

PyTorch implementation of function approximator. It needs three arguments to 
initialize: `PytorchNet` object, loss and optimizer. Loss and optimizers can be
imported directly from PyTorch. `PytorchNet` class is similar to `torch.nn.Module`
but with additional method `predict`.

#### PytorchNet

`prl.function_approximators.pytorch_nn`

Some neural networks and losses implementations used for RL problems
are kept in this module.

## Callbacks

`prl.callbacks`

You can pass some callbacks to the agent `train` method to control and supervise the training.
Some of the implemented callbacks are: `TensorboardLogger` to log training statistics to
tensorboardX, `EarlyStopping`, `PyTorchAgentCheckpoint`, `TrainingLogger`, `ValidationLogger`.

## Loggers and profiling

`prl.utils`

User and agents have access to five loggers. Most of them are used automatically. These
loggers are:

 * `time_logger` - this logger is used to monitor execution time of many functions and methods.
 You can print this object to generate report of execution times. If you want to profile
your function you can decorate your function with `prl.utils.timeit` decorator. From
now on, the execution time of this function will be logged.

 * `memory_logger` - logger is used to monitor RAM usage (currently unused).
 * `agent_logger` - this logger is used to monitor agent training statistics.
 * `nn_logger` - in this logger all the statistics from neural network training 
are stored. It is important to pass some distinct `id` argument to each network 
during initialization when training agent with many networks. This id will be 
used as a key in the logger.
 * `misc_logger` - logger for the user statistics. They are captured by the 
`TensorboardLogger` and plotted in the browser. You can log only numbers (ints or floats)
with a string key using `add` method.

## Agents

`prl.agents`

And finally the agents! Thanks to the above classes the agent implementations in
PRL are simple and compact. While implementing the agent all you need to do is
to implement `act`, `train_iteration` and `__init__` method. `train_iteration`
is a base step in agent training (e.g. one step in environment for DQN or 
some number of complete episodes in REINFORCE agent). You can also implement 
`pre_train_setup` and `post_train_cleanup` methods if needed. They are called
before and after main training loop.

`act` method is called by the agent while making one step in the environment. Agent 
have also methods inherited from base `Agent` class like: `play_episodes` and
`play_steps` and `test` which can be used within `train_iteration` method.

`train` method should be used only to initialize training from outside the agent.

Example agent code looks like this:

```python
class CrossEntropyAgent(Agent):
    """Agent using cross entropy algorithm"""

    def __init__(self, policy_network: FunctionApproximatorABC, agent_id: str = "crossentropy_agent"):
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

    def act(self, state: Observation) -> Action:
        state = state.reshape(1, *state.shape)
        act_probs = self.policy_network.predict(state)[0]
        return np.random.choice(len(act_probs), p=act_probs)
```

## Example

Let's take a look how to execute full RL experiment in PRL. After this tutorial
the code should be self-explanatory in most parts.

```python
import gym
import numpy as np
from torch import nn
from torch import optim
from prl.agents import CrossEntropyAgent
from prl.callbacks import TrainingLogger, EarlyStopping
from prl.environment import GymEnvironment
from prl.function_approximators import PytorchNN
from prl.function_approximators.pytorch_nn import PytorchMLP
from prl.transformers.reward_transformers import RewardShiftTransformer
from prl.transformers.state_transformers import StateShiftTransformer
from prl.utils import time_logger

gym_env = gym.make("CartPole-v0")
obs_shape = gym_env.observation_space.shape
state_transformer = StateShiftTransformer(np.zeros(obs_shape) - 0.5)
reward_transformer = RewardShiftTransformer(-0.5)

env = GymEnvironment(
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
policy_network = PytorchNN(net=net, loss=loss, optimizer=optimizer)
agent = CrossEntropyAgent(policy_network=policy_network)

callbacks = [
    EarlyStopping(target_reward=150, iteration_interval=5, number_of_test_runs=5),
    TrainingLogger(on_screen=True, iteration_interval=1),
]

agent.train(
    env, n_iterations=100, n_episodes=32, percentile=90, callback_list=callbacks
)

print(time_logger)
time_logger.save("./time_logger.pkl")
```

# Final remarks

The framework is under heavy development and can change in time. If you encounter
any problems with library, documentation or this tutorial please report it to us 
at piotr.tempczyk [at] opium.sh . If you want
to implement you agent and contribute to PRL we encourage you to do so. 
Just create a pull request :)
