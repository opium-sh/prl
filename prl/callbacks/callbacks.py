import glob
import os
import time
from copy import deepcopy
from functools import reduce
from math import gcd
from operator import iadd

import numpy as np
import torch
from tensorboardX import SummaryWriter

from prl.typing import AgentABC, HistoryABC, EnvironmentABC, AgentCallbackABC
from prl.utils import (
    timeit,
    time_logger,
    agent_logger,
    memory_logger,
    nn_logger,
    misc_logger,
)


class AgentCallback(AgentCallbackABC):
    """
    Interface for Callbacks defining actions that are executed automatically during
    different phases of agent training.
    """

    def __init__(self):
        self.time_logger_cid = time_logger.register()
        self.agent_logger_cid = agent_logger.register()
        self.memory_logger_cid = memory_logger.register()
        self.nn_logger_cid = nn_logger.register()
        self.misc_logger_cid = misc_logger.register()

    def on_iteration_end(self, agent: AgentABC) -> bool:
        """
        Method called at the end of every iteration in `prl.base.Agent.train` method.

        Args:
            agent: Agent in which this callback is called.

        Returns:
            True if training should be interrupted, False otherwise
        """

    def on_training_end(self, agent: AgentABC):
        """
        Method called after `prl.base.Agent.post_train_cleanup`.

        Args:
            agent: Agent in which this callback is called.
        """

    def on_training_begin(self, agent: AgentABC):
        """Method called after `prl.base.Agent.pre_train_setup`.

        Args:
            agent: Agent in which this callback is called
        """


class EarlyStopping(AgentCallback):
    """
    Implements EarlyStopping for RL Agents. Training is stopped after reaching given
    target reward.

    Args:
        target_reward: Target reward.
        iteration_interval:  Interval between calculating test reward.
            Using low values may make training process slower.
        number_of_test_runs: Number of test runs when calculating reward.
            Higher value averages variance out, but makes training longer.
        verbose: Whether to print message after stopping training (1), or not (0).

    Note:
        By reward, we mean here untransformed reward given by `Agent.test` method.
        For more info on methods see base class.
    """

    def __init__(
        self,
        target_reward: float,
        iteration_interval: int = 1,
        number_of_test_runs: int = 1,
        verbose: int = 1,
    ):
        super().__init__()
        self.number_of_test_runs = number_of_test_runs
        self.iteration_interval = iteration_interval
        self.target_reward = target_reward
        self.verbose = verbose
        self.needs_tests = True

    def on_iteration_end(self, agent: AgentABC):
        agent_logs = agent_logger.flush(self.agent_logger_cid)
        mean_test_reward = np.mean(agent_logs[0]["test_episode_total_reward"])
        break_flag = mean_test_reward >= self.target_reward
        if break_flag and self.verbose:
            print(
                "Early stopping in iteration_number %s. "
                "Achieved mean raw reward of %.4f (target was %.4f)"
                % (agent.iteration_count, mean_test_reward, self.target_reward)
            )
        return break_flag


class BaseAgentCheckpoint(AgentCallback):
    """
    Saving agents during training. This is a base class that implements only logic.
    One should use classes with saving method matching networks' framework.
    For more info on methods see base class.

    Args:
        target_path: Directory in which agents will be saved. Must exist before
        creating this callback.
        save_best_only: Whether to save all models, or only the one with highest reward.
        iteration_interval: Interval between calculating test reward. Using low values may make training process slower
        number_of_test_runs: Number of test runs when calculating reward. Higher value averages variance out, but makes training longer.
    """

    def __init__(
        self,
        target_path: str,
        save_best_only: bool = True,
        iteration_interval: int = 1,
        number_of_test_runs: int = 1,
    ):
        super().__init__()

        assert os.path.exists(target_path), (
            "Provided path (%s) does not exist!" % target_path
        )
        self.number_of_test_runs = number_of_test_runs
        self.iteration_interval = iteration_interval
        self.target_path = target_path
        self.save_best_only = save_best_only
        self.best_score = -np.inf
        self.needs_tests = True

    def _try_save(self, agent: AgentABC):
        agent_logs = agent_logger.flush(self.agent_logger_cid)
        mean_test_reward = np.mean(agent_logs[0]["test_episode_total_reward"])

        if mean_test_reward > self.best_score or not self.save_best_only:
            if mean_test_reward > self.best_score:
                self.best_score = mean_test_reward
            self._save_agent(agent, mean_test_reward)

    def on_iteration_end(self, agent: AgentABC):
        self._try_save(agent)

    def on_training_end(self, agent: AgentABC):
        self._try_save(agent)

    def _save_agent(self, agent: AgentABC, reward: float):
        raise NotImplementedError(
            "This is a base class for agent checkpoints. Please use one of "
            "the subclasses corresponding to your backend."
        )


class PyTorchAgentCheckpoint(BaseAgentCheckpoint):
    """Class for saving PyTorch-based agents. For more details, see parent class."""

    def _save_agent(self, agent: AgentABC, reward: float):
        if self.save_best_only:
            old_model_paths = glob.glob(os.path.join(self.target_path, agent.id + "*"))
        file_name = "%s_%s_%.4f" % (agent.id, agent.iteration_count, reward)
        full_path = os.path.join(self.target_path, file_name)
        torch.save(agent, full_path)
        if self.save_best_only:
            [os.remove(old_model_path) for old_model_path in old_model_paths]


class TrainingLogger(AgentCallback):
    """
    Logs training information after certain amount of iterations.
    Data may appear in output, or be written into a file.
    For more info on methods see base class.

    Args:
        on_screen: Whether to show info in output.
        to_file: Whether to save info into a file.
        file_path: Path to file with output.
        iteration_interval: How often should info be logged on screen. File output remains logged every iteration.
    """

    def __init__(
        self,
        on_screen: bool = True,
        to_file: bool = False,
        file_path: str = None,
        iteration_interval: int = 1,
    ):
        super().__init__()
        assert not to_file or (to_file and file_path is not None)
        self.on_screen = on_screen
        self.to_file = to_file
        self.file_path = file_path
        self.iteration_interval = iteration_interval
        self.needs_tests = False
        self._file_created = False
        self._last_agent_step_time = time.time()

    def _log_on_screen(
        self,
        iteration_number: int,
        mean_steps_per_second: float,
        loss: float,
        mean_reward: float,
        mean_episode_length: float,
    ):
        print(
            "Iteration: %d. "
            "Training metrics: loss=%.4f, total_reward_mean=%.2f, "
            "mean_episode_length=%.2f, mean_steps_per_second=%.4f, "
            % (
                iteration_number,
                loss,
                mean_reward,
                mean_episode_length,
                mean_steps_per_second,
            )
        )

    def _log_to_file(
        self,
        iteration_number: int,
        mean_steps_per_second: float,
        loss: float,
        mean_reward: float,
        mean_episode_length: float,
    ):
        mode = "w" if not self._file_created else "a"

        with open(self.file_path, mode) as f:
            if not self._file_created:
                f.write(
                    "iteration_number,loss,total_reward_mean,"
                    "mean_episode_length,mean_steps_per_second\n"
                )
                self._file_created = True

            f.write(
                "%s,%.4f,%.2f,%.2f,%.4f\n"
                % (
                    iteration_number,
                    loss,
                    mean_reward,
                    mean_episode_length,
                    mean_steps_per_second,
                )
            )

    def _calculate_summaries(self, agent_log, nn_log):
        # Sometimes agent_step is missing (rarely) - why?
        if "agent_step" in agent_log[2]:
            agent_step = agent_log[2]["agent_step"]
            # Include last step time from previous iteration to make this work
            # with just one step per iteration.
            agent_step = [self._last_agent_step_time] + agent_step
            mean_time_per_step = np.mean(np.diff(agent_step))
            mean_steps_per_second = 1 / mean_time_per_step
            self._last_agent_step_time = agent_step[-1]
        else:
            mean_steps_per_second = 0

        mean_loss_since_last_log = np.mean(list(nn_log[0].values())[0])
        # Episode might not have ended in this iteration.
        if "episode_total_reward" in agent_log[0]:
            mean_reward_since_last_log = np.mean(agent_log[0]["episode_total_reward"])
            mean_episode_length_since_last_log = np.mean(agent_log[0]["episode_length"])
        else:
            mean_reward_since_last_log = 0
            mean_episode_length_since_last_log = 0

        return (
            mean_steps_per_second,
            mean_loss_since_last_log,
            mean_reward_since_last_log,
            mean_episode_length_since_last_log,
        )

    def on_iteration_end(self, agent: AgentABC):
        recent_agent_logs = agent_logger.flush(self.agent_logger_cid)
        recent_nn_logs = nn_logger.flush(self.nn_logger_cid)

        summaries = self._calculate_summaries(recent_agent_logs, recent_nn_logs)

        if self.on_screen:
            self._log_on_screen(agent.iteration_count, *summaries)

        if self.to_file:
            self._log_to_file(agent.iteration_count, *summaries)


class ValidationLogger(AgentCallback):
    """
    Logs validation information after certain amount of iterations.
    Data may appear in output, or be written into a file.
    For more info on methods see base class.

    Args:
        on_screen: Whether to show info in output.
        to_file: Whether to save info into a file.
        file_path: Path to file with output.
        iteration_interval: How often should info be logged on screen. File output
        remains logged every iteration.
        number_of_test_runs: Number of played episodes in history's summary logs.
    """

    def __init__(
        self,
        on_screen: bool = True,
        to_file: bool = False,
        file_path: str = None,
        iteration_interval: int = 1,
        number_of_test_runs: int = 3,
    ):
        super().__init__()
        assert not to_file or (to_file and file_path is not None)
        self.on_screen = on_screen
        self.to_file = to_file
        self.file_path = file_path
        self.iteration_interval = iteration_interval
        self.number_of_test_runs = number_of_test_runs
        self.needs_tests = True

    def _log_on_screen(self, iteration_number: int, history_summary: tuple):
        print(
            "Iteration: %d. Validation metrics: total_reward_mean=%.1f, "
            "mean_length=%.1f" % ((iteration_number,) + history_summary)
        )

    def _log_to_file(self, iteration_number: int, history_summary: tuple):

        mode = "w" if iteration_number == 0 else "a"
        with open(self.file_path, mode) as f:
            if iteration_number == 0:
                f.write("iteration_number,total_reward_mean,mean_length\n")
            f.write("%s,%.1f,%.1f\n" % ((iteration_number,) + history_summary))

    def on_iteration_end(self, agent: AgentABC):
        agent_logs = agent_logger.flush(self.agent_logger_cid)
        mean_test_reward = np.mean(agent_logs[0]["test_episode_total_reward"])
        mean_test_episode_length = np.mean(agent_logs[0]["test_episode_length"])
        history_summary = (mean_test_reward, mean_test_episode_length)
        if self.on_screen:
            self._log_on_screen(agent.iteration_count, history_summary)

        if self.to_file:
            self._log_to_file(agent.iteration_count, history_summary)


class TensorboardLogger(AgentCallback):
    """
    Writes various information to tensorboard during training.
    For more info on methods see base class.

    Args:
        file_path: Path to file with output.
        iteration_interval: Interval between calculating test reward. Using low values may make training process slower.
        number_of_test_runs: Number of test runs when calculating reward. Higher value averages variance out, but makes training longer.
        show_time_logs: If shows logs from time_logger.
    """

    def __init__(
        self,
        file_path: str = "logs_" + str(int(time.time())),
        iteration_interval: int = 1,
        number_of_test_runs: int = 1,
        show_time_logs: bool = False,
    ):
        super().__init__()
        self.writer = SummaryWriter(file_path)
        self.iteration_interval = iteration_interval
        self.number_of_test_runs = number_of_test_runs
        self.show_time_logs = show_time_logs
        self.needs_tests = True

    def on_iteration_end(self, agent: AgentABC):
        names = ["agent_logger", "nn_logger", "memory_logger", "misc_logger"]
        loggers = [agent_logger, nn_logger, memory_logger, misc_logger]
        cids = [
            self.agent_logger_cid,
            self.nn_logger_cid,
            self.memory_logger_cid,
            self.misc_logger_cid,
        ]
        if self.show_time_logs:
            names.append("time_logger")
            loggers.append(time_logger)
            cids.append(self.time_logger_cid)
        for name, logger, cid in zip(names, loggers, cids):
            data, indicies, timestamps = logger.flush(cid)
            for key in data.keys():
                for i in range(len(data[key])):
                    self.writer.add_scalar(
                        "%s/%s" % (name, key),
                        data[key][i],
                        indicies[key][i],
                        timestamps[key][i],
                    )

    def on_training_end(self, agent: AgentABC):
        self.writer.close()


class CallbackHandler:
    """
    Callback that handles all given handles. Calls appropriate methods on each callback
    and aggregates break codes.
    For more info on methods see base class.
    """

    def __init__(self, callback_list: list, env: EnvironmentABC):
        self.callback_list = callback_list or []
        self.common_iteration_interval = None
        self.common_test_procedure_interval = None
        self.number_of_test_runs = None
        self.env = deepcopy(env)
        self.setup_callbacks()

    def setup_callbacks(self):
        """
        Sets up callbacks. This calculates optimal intervals for calling callbacks,
        and for calling testing procedure.
        """

        all_callback_iteration_intervals = (
            callback.iteration_interval for callback in self.callback_list
        )

        test_callback_iteration_intervals = (
            callback.iteration_interval
            for callback in self.callback_list
            if callback.needs_tests
        )

        numbers_of_callbacks_test_runs = (
            callback.number_of_test_runs
            for callback in self.callback_list
            if callback.needs_tests
        )

        # we have to run enough test iterations to satisfy callback with
        # highest amount of runs
        try:
            self.number_of_test_runs = max(numbers_of_callbacks_test_runs)
        except ValueError:
            # when there are no test runs we set number of test runs to 0
            self.number_of_test_runs = 0

        self.common_iteration_interval = reduce(
            gcd, all_callback_iteration_intervals, 0
        )
        self.common_test_procedure_interval = reduce(
            gcd, test_callback_iteration_intervals, 0
        )

    def run_tests(self, agent: AgentABC) -> HistoryABC:
        history_list = [agent.test(self.env) for _ in range(self.number_of_test_runs)]
        history = reduce(iadd, history_list)
        return history

    @staticmethod
    def check_run_condition(current_count, interval):
        if interval == 0:
            return False
        return current_count % interval == 0

    @timeit
    def on_iteration_end(self, agent: AgentABC):

        if not self.check_run_condition(
            agent.iteration_count, self.common_iteration_interval
        ):
            return False

        if self.check_run_condition(
            agent.iteration_count, self.common_test_procedure_interval
        ):
            self.run_tests(agent)

        break_signals = [
            callback.on_iteration_end(agent)
            for callback in self.callback_list
            if self.check_run_condition(
                agent.iteration_count, callback.iteration_interval
            )
        ]
        return any(break_signals)

    @timeit
    def on_training_end(self, agent: AgentABC):
        if self.number_of_test_runs > 0:
            self.run_tests(agent)
            [callback.on_training_end(agent) for callback in self.callback_list]

    @timeit
    def on_training_begin(self, agent: AgentABC):
        [callback.on_training_begin(agent) for callback in self.callback_list]
