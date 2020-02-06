from abc import abstractmethod
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from prl.function_approximators.function_approximators import FunctionApproximator
from prl.typing import PytorchNetABC
from prl.utils import timeit, nn_logger


class PytorchNet(PytorchNetABC):
    """
    Neural networks for PytorchFA. It has separate predict method strictly for
    Agent.act() method, wchich can act differently than forward() method.

    Note:
        This class has two abstract methods that need to be implemented (listed above).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Defines the computation performed at every training step.

        Args:
            x: input data

        Returns:
            network output
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor):
        """Makes prediction based on input data.

        Args:
            x: input data

        Returns:
            prediction for agent.act(x) method
        """
        pass


class PytorchFA(FunctionApproximator):
    """Class for pytorch based neural networks function approximators.

    Args:
        net: PytorchNet class neural network
        loss: loss function
        optimizer: optimizer
        device: device for computation: "cpu" or "cuda"
        batch_size: size of a training batch
        last_batch: flag if the last batch (usually shorter than batch_size) is going to be feed into network
        network_id: name of the network for debugging and logging purposes
    """

    def __init__(
        self,
        net: PytorchNet,
        loss: _Loss,
        optimizer: Optimizer,
        device: str = "cpu",
        batch_size: int = 64,
        last_batch: bool = True,
        network_id: str = "pytorch_nn",
    ):
        self._id = network_id
        self._device = device
        self._net = net.to(self._device)
        self._optimizer = optimizer
        self._loss = loss
        self._batch_size = batch_size
        self._last_batch = int(last_batch)

    def convert_to_pytorch(self, y: np.ndarray):
        if np.issubdtype(y.dtype, np.integer):
            y = torch.LongTensor(y).to(self._device)
        elif np.issubdtype(y.dtype, np.floating):
            y = torch.FloatTensor(y).to(self._device)
        return y

    @property
    def id(self):
        return self._id

    @timeit
    def train(self, x: np.ndarray, *loss_args):
        """Trains network on a dataset

        Args:
            x: input array for the network
            *loss_args: arguments passed directly to loss function
        """
        x = torch.from_numpy(x).to(self._device)
        indicies = torch.randperm(x.shape[0])
        loss_args = tuple(self.convert_to_pytorch(y) for y in loss_args)
        for i in range((indicies.shape[0] - 1) // self._batch_size + self._last_batch):
            start = i * self._batch_size
            end = (i + 1) * self._batch_size
            self._optimizer.zero_grad()
            batch = x[indicies[start:end]]
            y_pred = self._net(batch)
            loss_args_batch = (y[indicies[start:end]] for y in loss_args)
            loss = self._loss(y_pred, *loss_args_batch)
            loss.backward()
            nn_logger.add(self.id + str(id(self)), loss.item())
            self._optimizer.step()

    @timeit
    def predict(self, x: np.ndarray):
        """Makes prediction"""
        x = timeit(torch.from_numpy, "from_numpy")(x).float().to(self._device)
        return self._net.predict(x).cpu().data.numpy()


class PytorchMLP(PytorchNet):
    def __init__(self, x_shape, y_size, output_activation, hidden_sizes: Sequence[int]):
        super().__init__()
        assert len(x_shape) == 1, "Input must be flat for MLP."
        assert len(hidden_sizes) > 0
        self.y_size = y_size
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(x_shape[0], hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], y_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

    def predict(self, x):
        return self.output_activation(self.forward(x))


class PytorchConv(PytorchNet):
    def __init__(self, x_shape, hidden_sizes: Sequence[int], y_size):
        super().__init__()
        assert len(x_shape) == 3, "Input must be an image for a conv network."
        assert len(hidden_sizes) > 0
        self.softmax = nn.Softmax(dim=1)
        dims = [x_shape[-1]] + hidden_sizes
        (height, width, _) = x_shape
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(*in_out_dim, kernel_size=3, stride=2, padding=1)
                for in_out_dim in zip(dims[:-1], dims[1:])
            ]
        )
        self.out_layer = nn.Linear(
            height * width // 4 ** len(hidden_sizes) * dims[-1], y_size
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        for layer in self.conv_layers:
            x = F.relu(layer(x))
        return self.out_layer(x.view(x.size(0), -1))

    def predict(self, x):
        return self.softmax(self.forward(x))


class PolicyGradientLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super().__init__(size_average, reduce, reduction)

    def forward(self, nn_outputs, actions, returns):
        output_log_probs = F.log_softmax(nn_outputs, dim=1)
        log_prob_actions_v = returns * output_log_probs[range(len(actions)), actions]
        return -log_prob_actions_v.mean()


class DQNLoss(_Loss):
    def __init__(self, mode="huber", size_average=None, reduce=None, reduction="mean"):
        super().__init__(size_average, reduce, reduction)
        self.mode = mode
        self.loss = {"huber": F.smooth_l1_loss, "mse": F.mse_loss}[mode]

    def forward(self, nn_outputs, actions, target_outputs):
        target = nn_outputs.clone().detach()
        target[np.arange(target.shape[0]), actions] = target_outputs
        return self.loss(nn_outputs, target, reduction=self.reduction)
