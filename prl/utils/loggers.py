import pickle
from collections import deque, defaultdict
from numbers import Number
from time import perf_counter
from typing import Dict, Text, List

import numpy as np

from prl.utils.misc import colors

DEQUE_MAX_LEN = 10 ** 6


def limited_deque():
    """Auxiliary function for Logger class.

    Returns: Deque with maximum length set to DEQUE_MAX_LEN

    """
    return deque(maxlen=DEQUE_MAX_LEN)


class Logger:
    """
    Class for logging scalar values to limited queues. Logged data send to each
    client is tracked by the Logger, so each client can ask for unseen data and recieve it.
    """

    def __init__(self):
        self._data = defaultdict(limited_deque)
        self._timestamps = defaultdict(limited_deque)
        self._consumer_list = list()
        self._flush_indicies = dict()
        self._new_consumer_id = 0

    def add(self, key: str, value: Number):
        """Add a value to queue assigned to key value.

        Args:
            key: logged value name
            value: logged number
        """
        self._data[key].append(value)
        self._timestamps[key].append(perf_counter())

    def save(self, path: str):
        """Saves data to file.

        Args:
            path: path to the file.
        """
        pickle.dump(self._data, open(path, "wb"))
        self._clear()

    def register(self) -> int:
        """ Registers client in order to receive data from Logger object.

        Returns:
            client ID used to identify client while requesting for a new data.
        """
        consumer_id = self._new_consumer_id
        self._new_consumer_id += 1
        self._consumer_list.append(consumer_id)
        self._flush_indicies[consumer_id] = dict()
        return consumer_id

    def flush(
        self, consumer_id: int
    ) -> (Dict[str, List], Dict[str, range], Dict[str, List]):
        """Method used by clients to recieve only new unseed data from logger.

        Args:
            consumer_id: value returned by register method.

        Returns:
            dict with new data.

        """
        data_to_flush = dict()
        indicies_to_flush = dict()
        timestamps_to_flush = dict()
        for k in self._data:
            length = len(self._data[k])
            last_index = self._flush_indicies[consumer_id].get(k, 0)
            if last_index < length:
                data_to_flush[k] = list(self._data[k])[last_index:]
                indicies_to_flush[k] = tuple(range(last_index, length))
                timestamps_to_flush[k] = list(self._timestamps[k])[last_index:]
                self._flush_indicies[consumer_id][k] = length
        return data_to_flush, indicies_to_flush, timestamps_to_flush

    def get_data(self) -> Dict[str, deque]:
        """
        Returns:
            all logged data.

        """
        return self._data

    def _clear(self):
        """Clears the logged data"""
        self._data = defaultdict(limited_deque)
        self._timestamps = defaultdict(limited_deque)

    def __repr__(self):
        return "_data:\n%r\n_timestamps:\n%r\n_flush_indicies:\n%r\n" % (
            str(self._data),
            str(self._timestamps),
            str(self._flush_indicies),
        )


class TimeLogger(Logger):
    """
    Storage for measurements of function and methods exectuion time. Used by timeit function/decorator. Can be used to
    print summary of a time profiling or save all data to generate a plot how execution times are changing during the
    program execution.
    """

    def __str__(self):
        string = (
            "\n"
            + "_" * 27
            + "function_name_|__N_samp_|_mean_time"
            + "_" * 19
            + "|_total_time___\n"
        )
        format_str = "%s%40s | %7d | %12.4fms \xb1 %8.4fms | %10.2fs%s\n"
        for k, v in self._data.items():
            len_v = len(v)
            if v.maxlen > len_v:
                color = colors.END_FORMAT
            elif v.maxlen == len_v:
                color = colors.RED
            if len(v) > 1:
                string += format_str % (
                    color,
                    k,
                    len_v,
                    np.mean(list(v)[1:]) * 1000,
                    np.std(list(v)[1:]) * 1000,
                    np.sum(list(v)[1:]),
                    colors.END_FORMAT,
                )
            else:
                string += format_str % (
                    color,
                    k,
                    len_v,
                    np.mean(list(v)[:]) * 1000,
                    np.std(list(v)[:]) * 1000,
                    np.sum(list(v)[:]),
                    colors.END_FORMAT,
                )
        string += "_________________________________________|_________|_____________________________|______________\n\n"
        string += (
            "Rows with maximum length fo times buffer reached are marked in "
            + colors.RED
            + "RED.\n"
            + colors.END_FORMAT
        )
        return string


time_logger = TimeLogger()
memory_logger = Logger()
agent_logger = Logger()
nn_logger = Logger()
misc_logger = Logger()
