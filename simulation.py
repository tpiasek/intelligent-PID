"""
Closed-loop simulation: PID controller + plant model (see plant.py),
plus performance metrics / cost of a single step-response experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pid import PIDController
from plant import Plant, make_plant
from scenarios import Scenario


@dataclass(frozen=True)
class CostWeights:
    ise: float = 1.0
    control_energy: float = 0.025
    overshoot: float = 2.0       # applied to overshoot**2
    final_error: float = 2.0     # applied to final_error**2
    runaway_penalty: float = 20.0


@dataclass
class SimulationResult:
    """
    t       - Time vector [0, dt, ..., horizon - dt]
    y       - Plant output (process variable)
    u       - Plant input actually applied (controller output after the
              plant's own saturation, if it has one)
    error   - Setpoint - plant output
    metrics - {cost, ise, control_energy, overshoot, final_error}
    """
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray
    error: np.ndarray
    metrics: dict = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return self.metrics["cost"]


class ClosedLoopSimulator:
    """
    Runs step-response experiments of the PID + plant loop.

    :param dt: Time step [s]
    :param horizon: Length of an experiment [s]
    :param plant_model: "first_order" or "tanks" (see plant.PLANT_MODELS)
    :param plant_params: Extra keyword arguments for the plant constructor
                         (e.g. gain, delay, tb, y_nom, u_min, u_max)
    :param u_limit: Controller output saturation |u| <= u_limit
    :param min_step: Below this step size, metrics are normalised by min_step
    :param runaway_limit: |y| above this aborts the run (unstable gains)
    """

    def __init__(self,
                 dt: float = 0.02,
                 horizon: float = 12.0,
                 plant_model: str = "first_order",
                 plant_params: dict | None = None,
                 u_limit: float = 2.0,
                 min_step: float = 0.1,
                 runaway_limit: float = 5.0,
                 weights: CostWeights = CostWeights()):
        self.dt = dt
        self.horizon = horizon
        self.n_steps = int(horizon / dt)
        self.plant_model = plant_model
        self.plant_params = {"gain": 1.0, "delay": 0.5, **(plant_params or {})}
        self.make_plant(1.0)  # Validate model name / parameters early.
        self.u_limit = u_limit
        self.min_step = min_step
        self.runaway_limit = runaway_limit
        self.weights = weights

    def make_controller(self, gains) -> PIDController:
        return PIDController.from_gains(gains, dt=self.dt, u_limit=self.u_limit)

    def make_plant(self, tau: float) -> Plant:
        return make_plant(self.plant_model, tau=tau, dt=self.dt, **self.plant_params)

    def run(self, gains, scenario: Scenario) -> SimulationResult:
        """Simulate one step response for the given gains and scenario."""
        n = self.n_steps
        reference = scenario.reference
        t = np.arange(n) * self.dt
        y = np.zeros(n)
        u = np.zeros(n)
        error = np.zeros(n)

        plant = self.make_plant(scenario.tau)
        pid = self.make_controller(gains)
        y[0] = plant.reset(scenario.y0)
        pid.reset(initial_error=reference - y[0])

        for k in range(n - 1):
            error[k] = reference - y[k]
            u[k] = plant.saturate(pid.compute(error[k]))
            y_next = plant.step(u[k])
            y[k + 1] = y_next

            # Guard against numerical runaway during poor gain trials.
            if not np.isfinite(y_next) or abs(y_next) > self.runaway_limit:
                y[k + 1:] = np.sign(y[k]) * self.runaway_limit
                u[k + 1:] = 0.0
                error[k + 1:] = reference - y[k + 1:]
                break

        error[-1] = reference - y[-1]
        u[-1] = u[-2] if n > 1 else 0.0

        metrics = self.compute_metrics(y, u, error, scenario)
        return SimulationResult(t, y, u, error, metrics)

    def compute_metrics(self, y, u, error, scenario: Scenario) -> dict:
        """
        Error-based metrics are normalised by the step size |reference - y0|
        so that costs are comparable across scenarios with different steps.
        """
        dt, w = self.dt, self.weights
        reference = scenario.reference
        step = reference - scenario.y0
        scale = max(abs(step), self.min_step)
        direction = 1.0 if step >= 0 else -1.0

        ise = float(np.sum((error / scale) ** 2) * dt)
        control_energy = float(np.sum(u ** 2) * dt)
        overshoot = float(max(0.0, np.max(direction * (y - reference))) / scale)
        final_error = float(abs(reference - y[-1]) / scale)

        # Large penalty for trajectories that approach the runaway guard.
        runaway = np.max(np.abs(y)) >= self.runaway_limit - 0.1
        cost = (
            w.ise * ise
            + w.control_energy * control_energy
            + w.overshoot * overshoot ** 2
            + w.final_error * final_error ** 2
            + (w.runaway_penalty if runaway else 0.0)
        )
        return {
            "cost": float(cost),
            "ise": ise,
            "control_energy": control_energy,
            "overshoot": overshoot,
            "final_error": final_error,
        }
