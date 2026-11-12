"""
Operating scenarios and random generation of disjoint train / test datasets.

A *scenario* is a combination of
    - tau        : plant time constant,
    - y0         : starting condition (plant output, at rest, at t = 0),
    - reference  : setpoint the loop must reach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """One operating condition: plant time constant, start output, setpoint."""
    tau: float
    y0: float
    reference: float

    def label(self) -> str:
        return f"tau={self.tau:.2f}, y0={self.y0:+.2f}, r={self.reference:.2f}"

    def as_array(self) -> np.ndarray:
        return np.array([self.tau, self.y0, self.reference], dtype=float)


@dataclass(frozen=True)
class ScenarioSpace:
    """Uniform sampling ranges for each scenario parameter."""
    tau_range: tuple[float, float]
    y0_range: tuple[float, float]
    reference_range: tuple[float, float]
    min_step: float = 0.15  # Skip scenarios with |reference - y0| < min_step.

    def sample(self, rng: np.random.Generator, decimals: int = 3) -> Scenario:
        while True:
            tau = round(float(rng.uniform(*self.tau_range)), decimals)
            y0 = round(float(rng.uniform(*self.y0_range)), decimals)
            ref = round(float(rng.uniform(*self.reference_range)), decimals)
            if abs(ref - y0) >= self.min_step:
                return Scenario(tau, y0, ref)

    @property
    def widths(self) -> np.ndarray:
        return np.array([
            self.tau_range[1] - self.tau_range[0],
            self.y0_range[1] - self.y0_range[0],
            self.reference_range[1] - self.reference_range[0],
        ], dtype=float)


# Training: setpoints, starting conditions and time constants inside a core range.
# Includes both step-up (r > y0) and step-down (r < y0) cases.
DEFAULT_TRAIN_SPACE = ScenarioSpace(
    tau_range=(0.8, 1.2),
    y0_range=(-0.2, 0.6),
    reference_range=(0.3, 1.1),
)

# Testing: a wider range, so part of the test set lies outside the training
# range (extrapolation) and the rest between training samples (interpolation).
DEFAULT_TEST_SPACE = ScenarioSpace(
    tau_range=(0.75, 1.3),
    y0_range=(-0.25, 0.75),
    reference_range=(0.25, 1.2),
)

# Cascaded tanks: levels must be >= 0, so starting levels are non-negative.
# Here `tau` is the first tank's time constant Ta.
TANKS_TRAIN_SPACE = ScenarioSpace(
    tau_range=(0.8, 1.2),
    y0_range=(0.1, 0.8),
    reference_range=(0.3, 1.1),
)
TANKS_TEST_SPACE = ScenarioSpace(
    tau_range=(0.75, 1.3),
    y0_range=(0.0, 0.95),
    reference_range=(0.25, 1.2),
)

# Train / test spaces per plant model (keys match plant.PLANT_MODELS).
SCENARIO_SPACES = {
    "first_order": (DEFAULT_TRAIN_SPACE, DEFAULT_TEST_SPACE),
    "tanks": (TANKS_TRAIN_SPACE, TANKS_TEST_SPACE),
}


class ScenarioGenerator:
    """
    Randomly generates a training set and a disjoint test set.

    Every test scenario is at least `min_distance` away from every training
    scenario (Euclidean distance after normalising each parameter by the
    width of the training range), so the test set is truly unseen data and
    not just a near-duplicate of a training sample.
    """

    def __init__(self,
                 train_space: ScenarioSpace = DEFAULT_TRAIN_SPACE,
                 test_space: ScenarioSpace = DEFAULT_TEST_SPACE,
                 min_distance: float = 0.05):
        self.train_space = train_space
        self.test_space = test_space
        self.min_distance = min_distance
        self.rng = np.random.default_rng()
        self._scale = train_space.widths

    def _normalised(self, scenarios: list[Scenario]) -> np.ndarray:
        return np.array([s.as_array() for s in scenarios]) / self._scale

    def generate_train(self, n: int) -> list[Scenario]:
        scenarios: set[Scenario] = set()
        while len(scenarios) < n:
            scenarios.add(self.train_space.sample(self.rng))
        return sorted(scenarios, key=lambda s: (s.tau, s.y0, s.reference))

    def generate_test(self, n: int, train: list[Scenario],
                      max_attempts: int = 100_000) -> list[Scenario]:
        train_points = self._normalised(train) if train else np.empty((0, 3))
        scenarios: list[Scenario] = []
        attempts = 0
        while len(scenarios) < n:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "Could not generate enough test scenarios far enough from the "
                    "training set; lower min_distance or widen the test space."
                )
            s = self.test_space.sample(self.rng)
            if s in scenarios:
                continue
            if len(train_points):
                d = np.linalg.norm(train_points - s.as_array() / self._scale, axis=1)
                if d.min() < self.min_distance:
                    continue
            scenarios.append(s)
        return sorted(scenarios, key=lambda s: (s.tau, s.y0, s.reference))

    def generate(self, n_train: int, n_test: int) -> tuple[list[Scenario], list[Scenario]]:
        train = self.generate_train(n_train)
        test = self.generate_test(n_test, train)
        assert not set(train) & set(test), "Training and test scenarios must be disjoint"
        return train, test

    def is_extrapolation(self, s: Scenario) -> bool:
        """True if any parameter lies outside the training range."""
        sp = self.train_space
        return not (sp.tau_range[0] <= s.tau <= sp.tau_range[1]
                    and sp.y0_range[0] <= s.y0 <= sp.y0_range[1]
                    and sp.reference_range[0] <= s.reference <= sp.reference_range[1])
