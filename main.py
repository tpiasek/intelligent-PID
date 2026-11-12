"""
RL-based PID tuning on a simple nonlinear plant.

The SAC agent changes PID gains between complete step-response experiments.
The plant is nonlinear:
    dy/dt = (-y + 0.8*tanh(1.4*u) + 0.12*y**3) / tau

The cubic term is intentionally small; the input is saturated through tanh.
The environment randomizes tau and the reference slightly across episodes.
At each tuning step, the agent sees the current gains and performance metrics
from the previous experiment, then proposes bounded gain changes.

Run:
    pip install numpy scipy matplotlib gymnasium stable-baselines3
    python rl_pid_tuning_demo.py

This is a teaching demo, not a safety-certified controller.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# -------------------------- Plant and PID simulation --------------------------

DT = 0.02
HORIZON_SECONDS = 12.0
N_STEPS = int(HORIZON_SECONDS / DT)

# Gain bounds. These are deliberately modest for this example.
GAIN_LOW = np.array([0.0, 0.0, 0.0], dtype=np.float32)
GAIN_HIGH = np.array([8.0, 4.0, 1.5], dtype=np.float32)


def simulate_closed_loop(
    gains: np.ndarray,
    tau: float = 1.0,
    reference: float = 0.7,
    dt: float = DT,
    horizon: float = HORIZON_SECONDS,
):
    """
    Simulation of closed loop system with PID controller and plant.
    Integral term is bounded to create anti-windup.
    Derivative term has low-pass filtering applied.

    :param gains: Kp, Ki, Kd gains {np.ndarray([Kp, Ki, Kd])}
    :param tau: ?
    :param reference: Setpoint {float}
    :param dt: Time step {float}
    :param horizon: Time of whole simulation {int}
    :return:
        t - Array from 1 to n where n = no. elements = horizon/dt
            {np.ndarray([1, ..., horizon/dt])}
        y - Plant's output - process_variable
        u - Controller's output - output
        error - (Setpoint (reference) - output (u))
        metrics = {cost, ise, control_energy, overshoot, final_error}
                cost - Calculated from all other metrics (summarizes all metrics)
                ise - Integral Squared Error
                control_energy - Cost of control, checks sum of u signal
                overshoot - How far is the highest altitude of y from reference (setpoint)
                final_error - How far is the final value of y from setpoint
    """
    """Run one step-response experiment and return signals + performance metrics."""
    kp, ki, kd = np.asarray(gains, dtype=float)
    n = int(horizon / dt)
    t = np.arange(n) * dt

    y = np.zeros(n)
    u = np.zeros(n)
    error = np.zeros(n)

    integral = 0.0
    previous_error = reference - y[0]
    raw_derivative = 0.0
    derivative_filter_alpha = 0.85

    for k in range(n - 1):
        error[k] = reference - y[k]

        P_out = kp * error[k]
        
        # Integral term, with anti-windup by bounding integral state (conditional integration).
        integral += error[k] * dt
        I_out = ki * np.clip(integral, -4.0, 4.0)

        # Derivative term, with simple low-pass filtering.
        raw_derivative = (error[k] - previous_error) / dt
        derivative_filtered = (
            derivative_filter_alpha * derivative_filtered
            + (1.0 - derivative_filter_alpha) * raw_derivative
        )
        D_out = kd * derivative_filtered

        # Controller's output calculation
        raw_u = P_out + I_out + D_out
        u[k] = np.clip(raw_u, -2.0, 2.0)

        # # Nonlinear plant dynamics. Saturating input + weak cubic output term.
        # dydt = (-y[k] + 0.8 * np.tanh(1.4 * u[k]) + 0.12 * y[k] ** 3) / tau
        # y[k + 1] = y[k] + dt * dydt

        # First-order plant with transport delay:
        # T * dy/dt + y(t) = K * u(t - delay)

        T = 1.0  # Time constant [s]
        K = 1.0  # Static gain
        delay = 0.5  # Transport delay [s]

        delay_steps = int(round(delay / dt))

        # Retrieve the delayed control signal.
        delayed_index = k - delay_steps
        if delayed_index >= 0:
            u_delayed = u[delayed_index]
        else:
            u_delayed = 0.0

        # Differential equation:
        # dy/dt = (-y + K*u_delayed) / T
        dydt = (-y[k] + K * u_delayed) / T

        # Numerical integration (forward Euler).
        y[k + 1] = y[k] + dt * dydt

        # Guard against numerical runaway during poor gain trials.
        if not np.isfinite(y[k + 1]) or abs(y[k + 1]) > 5.0:
            y[k + 1:] = np.sign(y[k]) * 5.0
            u[k + 1:] = 0.0
            error[k + 1:] = reference - y[k + 1:]
            break

        previous_error = error[k]

    error[-1] = reference - y[-1]
    u[-1] = u[-2] if n > 1 else 0.0

    # Normalised terms keep reward magnitudes reasonably well-behaved.
    ise = float(np.sum(error**2) * dt)
    control_energy = float(np.sum(u**2) * dt)
    overshoot = float(max(0.0, np.max(y) - reference))
    final_error = float(abs(reference - y[-1]))

    # Large penalty for trajectories that approach the runaway guard.
    runaway_penalty = 20.0 if np.max(np.abs(y)) >= 4.9 else 0.0
    cost = (
        1.0 * ise
        + 0.025 * control_energy
        + 2.0 * overshoot**2
        + 2.0 * final_error**2
        + runaway_penalty
    )

    metrics = {
        "cost": cost,
        "ise": ise,
        "control_energy": control_energy,
        "overshoot": overshoot,
        "final_error": final_error,
    }
    return t, y, u, error, metrics


# ----------------------------- RL environment -----------------------------

class PIDTuningEnv(gym.Env):
    """
    Each action adjusts the PID gains, then the environment runs a full
    step-response experiment. The observation contains gains and metrics from
    the last trial. The reward favours lower cost and improvement over the
    previous trial.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_trials: int = 12):
        super().__init__()
        self.max_trials = max_trials

        # Actions are relative gain changes in [-1, 1].
        self.action_space = spaces.Box(

            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # [normalized Kp, Ki, Kd, ISE, overshoot, final error, control energy,
        #  tau, reference, previous normalized cost]
        obs_low = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0.6, 0.55, 0], dtype=np.float32
        )
        obs_high = np.array(
            [1, 1, 1, 10, 5, 5, 20, 1.4, 0.85, 10], dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        self.rng = np.random.default_rng()
        self.gains = np.array([1.0, 0.3, 0.05], dtype=np.float32)
        self.tau = 1.0
        self.reference = 0.7
        self.trial = 0
        self.previous_cost = None
        self.last_metrics = None

    def _observation(self):
        m = self.last_metrics or {
            "ise": 0.0, "overshoot": 0.0, "final_error": 0.0,
            "control_energy": 0.0, "cost": 0.0
        }
        return np.array([
            self.gains[0] / GAIN_HIGH[0],
            self.gains[1] / GAIN_HIGH[1],
            self.gains[2] / GAIN_HIGH[2],
            np.clip(m["ise"], 0, 10),
            np.clip(m["overshoot"], 0, 5),
            np.clip(m["final_error"], 0, 5),
            np.clip(m["control_energy"], 0, 20),
            self.tau,
            self.reference,
            np.clip(m["cost"], 0, 10),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.tau = float(self.rng.uniform(0.75, 1.25))
        self.reference = float(self.rng.uniform(0.60, 0.80))
        self.trial = 0

        # Randomise starting gains, so the policy sees varied tuning situations.
        self.gains = np.array([
            self.rng.uniform(0.4, 3.0),
            self.rng.uniform(0.05, 5),
            self.rng.uniform(0.0, 0.25),
        ], dtype=np.float32)
        self.gains = np.clip(self.gains, GAIN_LOW, GAIN_HIGH)                               # Change clipping to scaling [GAIN_LOW, GAIN_HIGH]

        _, _, _, _, metrics = simulate_closed_loop(
            self.gains, self.tau, self.reference
        )
        self.last_metrics = metrics
        self.previous_cost = metrics["cost"]
        return self._observation(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        # Scale actions to modest changes in each gain per trial.
        step_sizes = np.array([1.0, 0.45, 0.15])
        self.gains = np.clip(
            self.gains + action * step_sizes, GAIN_LOW, GAIN_HIGH
        )

        _, _, _, _, metrics = simulate_closed_loop(
            self.gains, self.tau, self.reference
        )

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


# -------------------------------- Main -----------------------------------

def main():
    np.random.seed(7)

    env = Monitor(PIDTuningEnv(max_trials=12))
    callback = EpisodeLogger()

    # CPU training is sufficient for this small model.
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-3,
        buffer_size=30_000,
        learning_starts=500,
        batch_size=128,
        tau=0.02,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
        device="cpu",
        seed=7,
    )

    # Increase to 20_000-50_000 for more training; this demo uses 12_000.
    total_timesteps = 10_000
    print(f"Training SAC for {total_timesteps:,} tuning steps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("sac_pid_tuner")
    print("Saved trained policy to sac_pid_tuner.zip")

    # Compare initial and learned tuning on the same nominal plant.
    tau_test = 1.0
    reference_test = 0.7
    initial_gains = np.array([1.0, 0.3, 0.05], dtype=np.float32)

    # Evaluate policy in a fresh environment. At each trial, it sees the
    # current gains and previous trial's metrics, then proposes gain changes.
    eval_env = PIDTuningEnv(max_trials=12)
    obs, _ = eval_env.reset(seed=123)
    eval_env.tau = tau_test
    eval_env.reference = reference_test
    _, _, _, _, first_metrics = simulate_closed_loop(
        initial_gains, tau_test, reference_test
    )
    eval_env.gains = initial_gains.copy()
    eval_env.last_metrics = first_metrics
    eval_env.previous_cost = first_metrics["cost"]
    obs = eval_env._observation()

    eval_gains = [initial_gains.copy()]
    eval_costs = [first_metrics["cost"]]
    for _ in range(12):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        eval_gains.append(eval_env.gains.copy())
        eval_costs.append(info["cost"])
        if terminated or truncated:
            break

    tuned_gains = eval_env.gains.copy()

    t, y_initial, u_initial, _, metrics_initial = simulate_closed_loop(
        initial_gains, tau_test, reference_test
    )
    _, y_tuned, u_tuned, _, metrics_tuned = simulate_closed_loop(
        tuned_gains, tau_test, reference_test
    )

    print("\nNominal plant comparison")
    print(f"Initial gains: Kp={initial_gains[0]:.3f}, "
          f"Ki={initial_gains[1]:.3f}, Kd={initial_gains[2]:.3f}")
    print(f"Tuned gains:   Kp={tuned_gains[0]:.3f}, "
          f"Ki={tuned_gains[1]:.3f}, Kd={tuned_gains[2]:.3f}")
    print(f"Initial cost: {metrics_initial['cost']:.4f}")
    print(f"Tuned cost:   {metrics_tuned['cost']:.4f}")
    print("Note: RL is stochastic and this short training run is illustrative; "
          "it may not outperform the initial gains every run.")

    # ----------------------------- Visualisation -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("SAC agent learning to tune a PID controller", fontsize=15)

    # 1. Training episode reward.
    ax = axes[0, 0]
    ep_rewards = np.asarray(callback.episode_rewards)
    ax.plot(np.arange(1, len(ep_rewards) + 1), ep_rewards, alpha=0.35, label="Episode reward")
    if len(ep_rewards) >= 10:
        window = min(20, max(3, len(ep_rewards) // 10))
        smooth = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, window + len(smooth)), smooth, label=f"{window}-episode mean")
    ax.set_title("Training reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Gain changes over sequential evaluation trials.
    ax = axes[0, 1]
    gh = np.asarray(eval_gains)
    ax.plot(range(len(gh)), gh[:, 0], marker="o", label="Kp")
    ax.plot(range(len(gh)), gh[:, 1], marker="o", label="Ki")
    ax.plot(range(len(gh)), gh[:, 2], marker="o", label="Kd")
    ax.set_title("Agent's gain adjustments (evaluation)")
    ax.set_xlabel("Tuning trial")
    ax.set_ylabel("Gain value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Response comparison.
    ax = axes[1, 0]
    ax.plot(t, np.full_like(t, reference_test), "--", label="Setpoint")
    ax.plot(t, y_initial, label="Initial PID")
    ax.plot(t, y_tuned, label="RL-tuned PID")
    ax.set_title("Step response on the same nonlinear plant")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output y(t)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Cost across evaluation tuning trials.
    ax = axes[1, 1]
    ax.plot(range(len(eval_costs)), eval_costs, marker="o")
    ax.set_title("Cost after each tuning experiment")
    ax.set_xlabel("Tuning trial")
    ax.set_ylabel("Cost (lower is better)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("rl_pid_tuning_results.png", dpi=160, bbox_inches="tight")
    print("\nSaved plot to rl_pid_tuning_results.png")
    plt.show()


if __name__ == "__main__":
    main()
