"""
RL-based PID tuning (SAC) with generalisation to unseen operating scenarios.

The SAC agent changes PID gains between complete step-response experiments.
Two plant models are available (--plant), see plant.py:

    first_order : tau * dy/dt + y(t) = K * u(t - delay)
    tanks       : two cascaded tanks, nonlinear 2nd order with saturation:
                  Ta * dy1/dt = K * sat(u)(t - delay) - 2 * sqrt(y_nom * y1)
                  Tb * dy2/dt = y1 - y2,     y = y2

Training and test scenarios (tau, y0, reference) are generated RANDOMLY. The
test set is sampled from a wider range than the training set and every test
scenario is kept a minimum distance away from all training scenarios, so the
two datasets are disjoint and the test measures generalisation (interpolation
and extrapolation) rather than memorisation.

Run (from this directory):
    pip install numpy scipy matplotlib gymnasium stable-baselines3
    python train_test.py [--plant tanks] [--timesteps 20000] [--no-show]

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from dataclasses import dataclass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib.pyplot as plt

from agent import EpisodeLogger, SACPIDTuner
from pid import NOMINAL_INITIAL_GAINS
from plant import PLANT_MODELS
from scenarios import SCENARIO_SPACES, Scenario, ScenarioGenerator
from simulation import ClosedLoopSimulator


# ------------------------------ Evaluation -------------------------------

class Evaluator:
    """Runs the trained agent on a set of scenarios and summarises results."""

    def __init__(self, tuner: SACPIDTuner, initial_gains=NOMINAL_INITIAL_GAINS):
        self.tuner = tuner
        self.initial_gains = np.asarray(initial_gains, dtype=np.float32)

    def evaluate(self, scenarios: list[Scenario], name: str,
                 verbose: bool = True, generator: ScenarioGenerator | None = None) -> list[dict]:
        results = [self.tuner.tune(s, self.initial_gains) for s in scenarios]
        initial = np.array([r["initial_cost"] for r in results])
        tuned = np.array([r["tuned_cost"] for r in results])

        if verbose:
            print(f"\n=== {name}: {len(results)} scenarios ===")
            print(f"{'scenario':<38}{'init cost':>10}{'tuned cost':>12}"
                  f"{'Kp':>8}{'Ki':>8}{'Kd':>8}")
            for r in results:
                kp, ki, kd = r["tuned_gains"]
                tag = " *" if generator and generator.is_extrapolation(r["scenario"]) else "  "
                print(f"{r['scenario'].label() + tag:<38}{r['initial_cost']:>10.3f}"
                      f"{r['tuned_cost']:>12.3f}{kp:>8.2f}{ki:>8.2f}{kd:>8.2f}")
            if generator:
                print("(* = outside the training range, i.e. extrapolation)")

        print(f"\n{name} summary: mean initial cost {initial.mean():.3f}, "
              f"mean tuned cost {tuned.mean():.3f} "
              f"(cost change {100 * (tuned.mean() / initial.mean() - 1):+.1f}%), "
              f"improved in {np.mean(tuned < initial) * 100:.0f}% of scenarios")
        return results


# ------------------------------ Visualisation ----------------------------

class ResultPlotter:
    def __init__(self, simulator: ClosedLoopSimulator, out_dir: Path):
        self.simulator = simulator
        self.out_dir = out_dir

    def plot(self, logger: EpisodeLogger, train_results, test_results,
             train_scenarios, test_scenarios, show: bool = True) -> None:
        self._plot_summary(logger, train_results, test_results)
        self._plot_datasets(train_scenarios, test_scenarios)
        self._plot_step_responses(test_results)
        if show:
            plt.show()

    def _save(self, fig, name: str) -> None:
        path = self.out_dir / name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"Saved plot to {path}")

    def _plot_summary(self, logger, train_results, test_results) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle("SAC PID tuner: training vs unseen test scenarios", fontsize=15)

        # 1. Training episode reward.
        ax = axes[0, 0]
        ep_rewards = np.asarray(logger.episode_rewards)
        ax.plot(np.arange(1, len(ep_rewards) + 1), ep_rewards, alpha=0.35, label="Episode reward")
        if len(ep_rewards) >= 10:
            window = min(20, max(3, len(ep_rewards) // 10))
            smooth = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window, window + len(smooth)), smooth, label=f"{window}-episode mean")
        ax.set_title("Training reward (random training scenarios)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative reward")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 2. Mean cost across tuning trials, train vs test.
        ax = axes[0, 1]
        for results, name in ((train_results, "Train scenarios"), (test_results, "Test scenarios")):
            costs = np.stack([r["cost_history"] for r in results])
            mean, std = costs.mean(axis=0), costs.std(axis=0)
            trials = np.arange(len(mean))
            ax.plot(trials, mean, marker="o", label=name)
            ax.fill_between(trials, mean - std, mean + std, alpha=0.15)
        ax.set_title("Mean cost during evaluation tuning (± 1 std)")
        ax.set_xlabel("Tuning trial")
        ax.set_ylabel("Cost (lower is better)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 3. Initial vs tuned cost per scenario.
        ax = axes[1, 0]
        for results, name, marker in ((train_results, "Train", "o"), (test_results, "Test", "^")):
            ax.scatter([r["initial_cost"] for r in results],
                       [r["tuned_cost"] for r in results],
                       marker=marker, alpha=0.7, label=name)
        all_costs = [c for r in train_results + test_results
                     for c in (r["initial_cost"], r["tuned_cost"])]
        lim = (0, max(all_costs) * 1.05)
        ax.plot(lim, lim, "k--", linewidth=1, label="No improvement")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title("Per-scenario cost (below diagonal = improved)")
        ax.set_xlabel("Initial cost")
        ax.set_ylabel("Tuned cost")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 4. Tuned gains vs setpoint on the test scenarios.
        ax = axes[1, 1]
        refs = np.array([r["scenario"].reference for r in test_results])
        tuned = np.stack([r["tuned_gains"] for r in test_results])
        for i, gname in enumerate(("Kp", "Ki", "Kd")):
            ax.scatter(refs + (i - 1) * 0.015, tuned[:, i], alpha=0.7, label=gname)
        ax.set_title("Tuned gains on test scenarios")
        ax.set_xlabel("Setpoint (reference)")
        ax.set_ylabel("Gain value")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        self._save(fig, "rl_pid_tuning_results.png")

    def _plot_datasets(self, train_scenarios, test_scenarios) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle("Randomly generated datasets (train vs test)", fontsize=14)
        pairs = (("y0", "reference"), ("tau", "reference"), ("tau", "y0"))
        for ax, (xa, ya) in zip(axes, pairs):
            for data, name, marker in ((train_scenarios, "Train", "o"),
                                       (test_scenarios, "Test", "^")):
                ax.scatter([getattr(s, xa) for s in data], [getattr(s, ya) for s in data],
                           marker=marker, alpha=0.7, label=name)
            ax.set_xlabel(xa)
            ax.set_ylabel(ya)
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.tight_layout()
        self._save(fig, "rl_pid_datasets.png")

    def _plot_step_responses(self, test_results) -> None:
        picks = np.linspace(0, len(test_results) - 1, min(6, len(test_results))).round().astype(int)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        fig.suptitle("Step responses on unseen test scenarios", fontsize=15)
        for ax, idx in zip(axes.flat, picks):
            r = test_results[idx]
            s = r["scenario"]
            init = self.simulator.run(r["initial_gains"], s)
            tuned = self.simulator.run(r["tuned_gains"], s)
            ax.plot(init.t, np.full_like(init.t, s.reference), "--", color="gray", label="Setpoint")
            ax.plot(init.t, init.y, label=f"Initial PID (cost {r['initial_cost']:.2f})")
            ax.plot(tuned.t, tuned.y, label=f"RL-tuned PID (cost {r['tuned_cost']:.2f})")
            ax.set_title(s.label(), fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Time (s)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Output y(t)")
        fig.tight_layout()
        self._save(fig, "rl_pid_test_responses.png")


# -------------------------------- Main -----------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plant", choices=list(PLANT_MODELS), default="first_order",
                        help="Plant model to control")
    parser.add_argument("--timesteps", type=int, default=10_000,
                        help="SAC training steps (each step = one closed-loop experiment)")
    parser.add_argument("--n-train", type=int, default=60, help="Number of training scenarios")
    parser.add_argument("--n-test", type=int, default=40, help="Number of test scenarios")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows")
    return parser.parse_args()

@dataclass
class Config:
    plant: str = "tanks"
    timesteps: int = 20_000
    n_train: int = 60
    n_test: int = 40
    show: bool = True


def main():
    cfg = Config()
    out_dir = Path(__file__).parent / "results" / cfg.plant
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plant: {cfg.plant}")

    # 1. Random, disjoint training and test datasets.
    train_space, test_space = SCENARIO_SPACES[cfg.plant]
    generator = ScenarioGenerator(train_space, test_space)
    train_scenarios, test_scenarios = generator.generate(cfg.n_train, cfg.n_test)
    n_extrap = sum(generator.is_extrapolation(s) for s in test_scenarios)
    print(f"{len(train_scenarios)} training scenarios, {len(test_scenarios)} held-out "
          f"test scenarios ({n_extrap} outside the training range)")

    # 2. Train the agent on the training scenarios only.
    simulator = ClosedLoopSimulator(plant_model=cfg.plant)
    tuner = SACPIDTuner(simulator=simulator, max_trials=12)
    model_path = out_dir / "sac_pid_tuner"
    # More scenarios need more experience; increase to 50_000+ for better results.
    print(f"Training SAC for {cfg.timesteps:,} tuning steps...")
    tuner.train(train_scenarios, total_timesteps=cfg.timesteps)
    tuner.save(str(model_path))
    print(f"Saved trained policy to {model_path}.zip")

    # 3. Test: tune from the same nominal gains on (a sample of) the training
    #    scenarios and on all unseen test scenarios -> generalisation gap.
    rng = np.random.default_rng()
    sample_idx = rng.choice(len(train_scenarios),
                            size=min(len(test_scenarios), len(train_scenarios)), replace=False)
    train_sample = [train_scenarios[i] for i in sorted(sample_idx)]

    evaluator = Evaluator(tuner, NOMINAL_INITIAL_GAINS)
    train_results = evaluator.evaluate(train_sample, "TRAIN scenarios (seen)", verbose=False)
    test_results = evaluator.evaluate(test_scenarios, "TEST scenarios (unseen)",
                                      verbose=True, generator=generator)
    print("Note: RL is stochastic and this short training run is illustrative; "
          "it may not outperform the initial gains in every scenario.")

    # 4. Plots.
    ResultPlotter(simulator, out_dir).plot(
        tuner.logger, train_results, test_results,
        train_scenarios, test_scenarios, show=cfg.show,
    )


if __name__ == "__main__":
    main()
