"""
RL side of the PID tuner:
    - PIDTuningEnv  : gymnasium environment, one step = one closed-loop experiment,
    - EpisodeLogger : SB3 callback collecting training history,
    - SACPIDTuner   : wrapper around a stable-baselines3 SAC agent.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from pid import GAIN_HIGH, GAIN_LOW
from scenarios import Scenario
from simulation import ClosedLoopSimulator


# ----------------------------- RL environment -----------------------------

class PIDTuningEnv(gym.Env):
    """
    Each episode draws a scenario (tau, y0, reference) from `scenarios` and
    random initial gains. Each action adjusts the PID gains, then the
    environment runs a full step-response experiment. The observation contains
    the gains, the metrics from the last trial and the scenario description.
    The reward favours lower cost and improvement over the previous trial.

    reset(options=...) accepts:
        "scenario": Scenario  - force a specific scenario (used for testing)
        "gains":    array(3)  - force specific initial gains
    """

    metadata = {"render_modes": []}

    # Scale of actions -> gain change per trial.
    GAIN_STEP_SIZES = np.array([1.0, 0.45, 0.15])

    def __init__(self, scenarios: list[Scenario],
                 simulator: ClosedLoopSimulator | None = None,
                 max_trials: int = 12):
        super().__init__()
        if not scenarios:
            raise ValueError("At least one scenario is required")
        self.scenarios = list(scenarios)
        self.simulator = simulator or ClosedLoopSimulator()
        self.max_trials = max_trials

        # Actions are relative gain changes in [-1, 1].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # [normalized Kp, Ki, Kd, ISE, overshoot, final error, control energy,
        #  tau, y0, reference, previous cost]
        obs_low = np.array([0, 0, 0, 0, 0, 0, 0, 0.5, -1.0, -1.0, 0], dtype=np.float32)
        obs_high = np.array([1, 1, 1, 10, 5, 5, 20, 1.5, 2.0, 2.0, 10], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.rng = np.random.default_rng()
        self.scenario = self.scenarios[0]
        self.gains = np.zeros(3, dtype=np.float32)
        self.trial = 0
        self.previous_cost = None
        self.last_metrics = None

    def _simulate(self) -> dict:
        return self.simulator.run(self.gains, self.scenario).metrics

    def _observation(self) -> np.ndarray:
        m = self.last_metrics or {
            "ise": 0.0, "overshoot": 0.0, "final_error": 0.0,
            "control_energy": 0.0, "cost": 0.0,
        }
        return np.array([
            self.gains[0] / GAIN_HIGH[0],
            self.gains[1] / GAIN_HIGH[1],
            self.gains[2] / GAIN_HIGH[2],
            np.clip(m["ise"], 0, 10),
            np.clip(m["overshoot"], 0, 5),
            np.clip(m["final_error"], 0, 5),
            np.clip(m["control_energy"], 0, 20),
            self.scenario.tau,
            self.scenario.y0,
            self.scenario.reference,
            np.clip(m["cost"], 0, 10),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        self.trial = 0

        if "scenario" in options:
            self.scenario = options["scenario"]
        else:
            self.scenario = self.scenarios[self.rng.integers(len(self.scenarios))]

        if "gains" in options:
            self.gains = np.asarray(options["gains"], dtype=np.float32).copy()
        else:
            # Randomise starting gains, so the policy sees varied tuning situations.
            self.gains = np.array([
                self.rng.uniform(0.4, 3.0),
                self.rng.uniform(0.05, 5),
                self.rng.uniform(0.0, 0.25),
            ], dtype=np.float32)
        self.gains = np.clip(self.gains, GAIN_LOW, GAIN_HIGH)

        metrics = self._simulate()
        self.last_metrics = metrics
        self.previous_cost = metrics["cost"]
        return self._observation(), {"scenario": self.scenario}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.gains = np.clip(
            self.gains + action * self.GAIN_STEP_SIZES, GAIN_LOW, GAIN_HIGH
        ).astype(np.float32)

        metrics = self._simulate()

        # Reward combines absolute performance and improvement.
        # Improvement term makes it easier to learn which adjustments help.
        improvement = self.previous_cost - metrics["cost"]
        reward = -0.20 * metrics["cost"] + 0.35 * improvement
        reward = float(np.clip(reward, -10.0, 10.0))

        self.previous_cost = metrics["cost"]
        self.last_metrics = metrics
        self.trial += 1

        terminated = self.trial >= self.max_trials
        truncated = False
        info = {
            **metrics,
            "kp": float(self.gains[0]),
            "ki": float(self.gains[1]),
            "kd": float(self.gains[2]),
            "trial": self.trial,
            "scenario": self.scenario,
        }
        return self._observation(), reward, terminated, truncated, info


# ------------------------------- Logging ---------------------------------

class EpisodeLogger(BaseCallback):
    """Collect episode rewards and per-trial tuning history for plots."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.gain_history = []
        self.cost_history = []
        self.reward_history = []
        self._current_episode_reward = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [])

        if rewards is not None:
            self._current_episode_reward += float(rewards[0])

        if infos:
            info = infos[0]
            if all(k in info for k in ("kp", "ki", "kd", "cost")):
                self.gain_history.append([info["kp"], info["ki"], info["kd"]])
                self.cost_history.append(info["cost"])
                self.reward_history.append(float(rewards[0]) if rewards is not None else 0.0)

        if dones is not None and bool(dones[0]):
            self.episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0

        return True


# -------------------------------- Agent ----------------------------------

class SACPIDTuner:
    """SAC agent that proposes PID gain changes between experiments."""

    DEFAULT_SAC_KWARGS = dict(
        learning_rate=3e-3,
        buffer_size=30_000,
        learning_starts=500,
        batch_size=128,
        tau=0.02,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [64, 64]},
        device="cpu",  # CPU is sufficient for this small model.
    )

    def __init__(self, simulator: ClosedLoopSimulator | None = None,
                 max_trials: int = 12,
                 verbose: int = 1, **sac_kwargs):
        self.simulator = simulator or ClosedLoopSimulator()
        self.max_trials = max_trials
        self.verbose = verbose
        self.sac_kwargs = {**self.DEFAULT_SAC_KWARGS, **sac_kwargs}
        self.model: SAC | None = None
        self.logger = EpisodeLogger()

    def make_env(self, scenarios: list[Scenario]) -> PIDTuningEnv:
        return PIDTuningEnv(scenarios, simulator=self.simulator, max_trials=self.max_trials)

    # --------------------------------------------------------------- training
    def train(self, scenarios: list[Scenario], total_timesteps: int) -> EpisodeLogger:
        """Train a fresh SAC model on the given (training) scenarios."""
        env = Monitor(self.make_env(scenarios))
        env.reset()
        self.model = SAC("MlpPolicy", env, verbose=self.verbose, **self.sac_kwargs)
        self.logger = EpisodeLogger()
        self.model.learn(total_timesteps=total_timesteps, callback=self.logger)
        return self.logger

    def save(self, path: str) -> None:
        self._require_model()
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = SAC.load(path, device=self.sac_kwargs.get("device", "auto"))

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("Agent has no model: call train() or load() first")

    # -------------------------------------------------------------- inference
    def tune(self, scenario: Scenario, initial_gains) -> dict:
        """Let the trained agent tune the PID for one scenario, from initial_gains."""
        self._require_model()
        env = self.make_env([scenario])
        obs, _ = env.reset(options={"scenario": scenario, "gains": initial_gains})

        gain_history = [env.gains.copy()]
        cost_history = [env.previous_cost]
        for _ in range(self.max_trials):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            gain_history.append(env.gains.copy())
            cost_history.append(info["cost"])
            if terminated or truncated:
                break

        return {
            "scenario": scenario,
            "initial_gains": np.asarray(initial_gains, dtype=np.float32).copy(),
            "tuned_gains": env.gains.copy(),
            "gain_history": np.asarray(gain_history),
            "cost_history": np.asarray(cost_history),
            "initial_cost": cost_history[0],
            "tuned_cost": cost_history[-1],
        }
