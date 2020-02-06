from unittest import TestCase

import numpy as np
from parameterized import parameterized

from prl.storage import Memory


class TestMemory(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = np.arange(0, 2 * 14, 1.0).reshape(14, 2)
        self.actions = np.arange(13).astype(np.int32)
        self.rewards = np.arange(13)
        self.dones = np.zeros(13).astype(np.bool)
        self.dones[6] = True

    def _assert_memory_empty(self, memory):
        for arr in [
            memory.get_states(),
            memory.get_actions(),
            memory.get_rewards(),
            memory.get_dones(),
        ]:
            self.assertEqual(len(arr), 0)

    def _assert_memory_contents(self, memory, from_index, to_index):
        for (actual_arr, expected_arr) in [
            (memory.get_states(), self.states),
            (memory.get_actions(), self.actions),
            (memory.get_rewards(), self.rewards),
            (memory.get_dones(), self.dones),
        ]:
            np.testing.assert_array_equal(actual_arr, expected_arr[from_index:to_index])

        np.testing.assert_array_equal(memory.get_last_state(), self.states[to_index])

    def _update_memory(self, memory, to_index):
        for i in range(to_index):
            memory.update(
                self.actions[i], self.rewards[i], self.dones[i], self.states[i + 1]
            )

    def test_init(self):
        memory = Memory(self.states[0], np.int32, 5)
        self._assert_memory_empty(memory)

    def test_new_state_update(self):
        memory = Memory(self.states[0], np.int32, 5)
        new_new_state = np.array([100.0, 100.0])
        memory.new_state_update(new_new_state)
        np.testing.assert_array_equal(memory.get_last_state(), new_new_state)
        self._assert_memory_empty(memory)

    @parameterized.expand(
        [
            ("num_{}_range_{}_{}".format(num, from_, to), num, from_, to)
            for (num, from_, to) in [(1, 0, 1), (7, 2, 7), (13, 8, 13)]
        ]
    )
    def test_add_SARDs(self, _, num, from_, to):
        memory = Memory(self.states[0], np.int32, 5)
        self._update_memory(memory, num)
        self._assert_memory_contents(memory, from_, to)

    def test_empty_after_clear(self):
        memory = Memory(self.states[0], np.int32, 5)
        self._update_memory(memory, 3)
        memory.clear(self.states[0])
        self._assert_memory_empty(memory)

    def test_add_one_frame_after_clear(self):
        memory = Memory(self.states[0], np.int32, 5)
        self._update_memory(memory, 1)
        memory.clear(self.states[0])
        self._update_memory(memory, 1)
        self._assert_memory_contents(memory, 0, 1)

    def test_include_last(self):
        memory = Memory(self.states[0], np.int32, 1)
        self._update_memory(memory, 1)
        np.testing.assert_array_equal(
            memory.get_states(include_last=True), self.states[:2]
        )
