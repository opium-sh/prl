from functools import wraps
from unittest import TestCase

import numpy as np
from parameterized import parameterized

from prl import agents


GAMMAS = (0.0, 0.5, 1.0)
GAMMA_PARAMETERS = [("gamma_{}".format(gamma), gamma) for gamma in GAMMAS]
LAMBDAS = (0.0, 0.5, 1.0)


def for_each_advantage(test_case):
    """Decorator for test cases, running them for all declared advantage functions."""

    @wraps(test_case)
    def wrapper(self, *args, **kwargs):
        for advantage in self.advantage_functions:
            test_case(self, advantage, *args, **kwargs)

    return wrapper


class TestAdvantageBase:
    """Base class for advantage tests.

    Tests properties common for all advantage functions.
    """

    advantage_functions = None

    def setUp(self):
        self.rewards = np.array([n ** 2 * (-1) ** n for n in range(5)])
        self.positive_rewards = np.abs(self.rewards)
        self.baselines = np.array([2, -1] * 3)
        self.positive_baselines = np.abs(self.baselines)
        self.dones = np.array([False] * 5)

    @staticmethod
    def perfect_baseline(rewards, discount_factor):
        baselines = np.zeros(len(rewards) + 1)
        baselines[-1] = 10
        for i in reversed(range(len(rewards))):
            baselines[i] = rewards[i] + discount_factor * baselines[i + 1]
        return baselines

    @for_each_advantage
    def test_ignores_future_when_gamma_zero(self, advantage):
        advantages = advantage(
            self.rewards, self.baselines, self.dones, discount_factor=0
        )
        np.testing.assert_array_equal(advantages, self.rewards - self.baselines[:-1])

    @for_each_advantage
    def test_sums_rewards_when_gamma_one(self, advantage):
        advantages = advantage(
            self.rewards, self.baselines, self.dones, discount_factor=1
        )
        self.assertEqual(
            advantages[0], np.sum(self.rewards) + self.baselines[-1] - self.baselines[0]
        )

    @parameterized.expand(GAMMA_PARAMETERS)
    @for_each_advantage
    def test_ignores_future_when_done(self, advantage, _, gamma):
        dones = np.array([False, False, True, False, False])
        advantages = advantage(
            self.rewards, self.baselines, dones, discount_factor=gamma
        )
        self.assertEqual(advantages[2], self.rewards[2] - self.baselines[2])

    @parameterized.expand(
        [
            ("increasing_when_rewards_and_baselines_positive", 1),
            ("decreasing_when_rewards_and_baselines_negative", -1),
        ]
    )
    @for_each_advantage
    def test_advantage_monotonic_with_gamma(self, advantage, _, multiplier):
        advantages_per_gamma = [
            advantage(
                self.positive_rewards * multiplier,
                self.positive_baselines * multiplier,
                self.dones,
                gamma,
            )
            for gamma in GAMMAS
        ]
        for (lower, upper) in zip(advantages_per_gamma[:-1], advantages_per_gamma[1:]):
            # Last element will be equal. Multiplier changes monotonicity.
            np.testing.assert_array_less(
                lower[:-1] * multiplier, upper[:-1] * multiplier
            )

    @parameterized.expand(GAMMA_PARAMETERS)
    @for_each_advantage
    def test_advantage_zero_when_perfect_baseline(self, advantage, _, gamma):
        baselines = self.perfect_baseline(self.rewards, gamma)
        advantages = advantage(self.rewards, baselines, self.dones, gamma)
        np.testing.assert_array_equal(advantages, np.zeros_like(advantages))


class TestA2CAdvantage(TestAdvantageBase, TestCase):
    advantage_functions = (agents.A2CAdvantage(),)


class TestGAEAdvantage(TestAdvantageBase, TestCase):
    advantage_functions = tuple(agents.GAEAdvantage(lambda_) for lambda_ in LAMBDAS)

    @parameterized.expand(GAMMA_PARAMETERS)
    def test_looks_one_step_when_lambda_zero(self, _, gamma):
        advantages = agents.GAEAdvantage(lambda_=0.0)(
            self.rewards, self.baselines, self.dones, gamma
        )
        np.testing.assert_array_equal(
            advantages, self.rewards + gamma * self.baselines[1:] - self.baselines[:-1]
        )

    # This property only holds for lambda = 1.
    def test_sums_rewards_when_gamma_one(self):
        super().test_sums_rewards_when_gamma_one.__wrapped__(
            self, agents.GAEAdvantage(lambda_=1.0)
        )

    def _test_advantage_monotonic_with_lambda(self, rewards, baselines, multiplier):
        advantages_per_lambda = [
            agents.GAEAdvantage(lambda_)(
                rewards * multiplier,
                baselines * multiplier,
                self.dones,
                discount_factor=1.0,
            )
            for lambda_ in LAMBDAS
        ]
        for (lower, upper) in zip(
            advantages_per_lambda[:-1], advantages_per_lambda[1:]
        ):
            np.testing.assert_array_less(
                lower[:-1] * multiplier, upper[:-1] * multiplier
            )

    @parameterized.expand(
        [
            ("increasing_when_baselines_increasing", 1),
            ("decreasing_when_baselines_decreasing", -1),
        ]
    )
    def test_advantage_monotonic_with_lambda(self, _, multiplier):
        zero_rewards = np.zeros_like(self.rewards)
        baselines_len = len(zero_rewards) + 1
        # Increasing but negative.
        increasing_baselines = np.arange(baselines_len) - baselines_len
        self._test_advantage_monotonic_with_lambda(
            zero_rewards, increasing_baselines, multiplier
        )

    @parameterized.expand(
        [
            ("increasing_when_rewards_positive", 1),
            ("decreasing_when_rewards_negative", -1),
        ]
    )
    def test_advantage_monotonic_with_lambda(self, _, multiplier):
        zero_baselines = np.zeros(len(self.positive_rewards) + 1)
        self._test_advantage_monotonic_with_lambda(
            self.positive_rewards, zero_baselines, multiplier
        )
