"""
Discrete PID controller.

    u = Kp * e + Ki * integral(e) + Kd * filtered(de/dt)

Features (identical to the original single-file version):
    - anti-windup by bounding the integral state,
    - first-order low-pass filter on the derivative term,
    - actuator saturation |u| <= u_limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Gain bounds [Kp, Ki, Kd].
GAIN_LOW = np.array([0.0, 0.0, 0.0], dtype=np.float32)
GAIN_HIGH = np.array([100.0, 100.0, 10.5], dtype=np.float32)

# Gains used as the common starting point when evaluating the trained agent.
NOMINAL_INITIAL_GAINS = np.array([1.0, 0.3, 0.05], dtype=np.float32)


@dataclass
class PIDController:
    """
    :param kp, ki, kd: PID gains
    :param dt: Sample time [s]
    :param u_limit: Actuator saturation, output is clipped to [-u_limit, u_limit]
    :param integral_limit: Bound on the integral state (anti-windup)
    :param derivative_filter_alpha: Low-pass coefficient for the derivative,
                                    0 = no filtering, close to 1 = heavy filtering
    """
    kp: float
    ki: float
    kd: float
    dt: float = 0.02
    u_limit: float = 2.0
    integral_limit: float = 4.0
    derivative_filter_alpha: float = 0.85

    def __post_init__(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.derivative_filtered = 0.0

    # ------------------------------------------------------------------ gains
    @property
    def gains(self) -> np.ndarray:
        return np.array([self.kp, self.ki, self.kd], dtype=float)

    @gains.setter
    def gains(self, value) -> None:
        self.kp, self.ki, self.kd = (float(g) for g in value)

    @classmethod
    def from_gains(cls, gains, **kwargs) -> "PIDController":
        kp, ki, kd = (float(g) for g in gains)
        return cls(kp, ki, kd, **kwargs)

    # ----------------------------------------------------------------- update
    def reset(self, initial_error: float = 0.0) -> None:
        """Clear internal state. `initial_error` avoids a derivative kick at k=0."""
        self.integral = 0.0
        self.previous_error = float(initial_error)
        self.derivative_filtered = 0.0

    def compute(self, error: float) -> float:
        """Return the saturated controller output for the current error."""
        dt = self.dt
        p_out = self.kp * error

        # Integral term, with anti-windup by bounding the integral state.
        lim = self.integral_limit
        self.integral = min(max(self.integral + error * dt, -lim), lim)
        i_out = self.ki * self.integral

        # Derivative term, with simple low-pass filtering.
        raw_derivative = (error - self.previous_error) / dt
        a = self.derivative_filter_alpha
        self.derivative_filtered = a * self.derivative_filtered + (1.0 - a) * raw_derivative
        d_out = self.kd * self.derivative_filtered

        self.previous_error = error

        # Saturated actuator.
        return min(max(p_out + i_out + d_out, -self.u_limit), self.u_limit)
