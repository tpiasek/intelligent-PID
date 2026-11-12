"""
Plant models (all integrated with forward Euler at a fixed time step).

1. FirstOrderDelayPlant - linear first-order inertia with transport delay:

       tau * dy/dt + y(t) = K * u(t - delay)

2. CascadeTanksPlant ("object CF") - nonlinear second-order inertia built as
   two cascaded tanks; the first tank's outflow is proportional to the square
   root of its level:

       Ta * dy1/dt = K * u_real(t - delay) - 2 * sqrt(y_nom * y1(t))
       Tb * dy2/dt = y1(t) - y2(t),          y(t) = y2(t)

       u_real(t) = sat(u(t)) = clip(u(t), u_min, u_max)

   y_nom is the nominal operating point. Linearised at y_nom (no delay, no
   saturation) the plant is the classic second-order inertia

       G(s) = K / (T2 s^2 + T1 s + 1),   T2 = Ta * Tb,   T1 = Ta + Tb.

   The nonlinearity comes from the saturation and the square-root outflow;
   the delay is linear but makes control harder.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque


class Plant(ABC):
    """
    Common interface: input saturation + transport delay line + dynamics.

    :param dt: Integration / sampling time step Ts [s]
    :param delay: Transport delay [s] (rounded to whole samples, see delay_steps)
    :param u_min, u_max: Input saturation; None means unbounded on that side
    """

    def __init__(self, dt: float = 0.02, delay: float = 0.5,
                 u_min: float | None = None, u_max: float | None = None):
        self.dt = float(dt)
        self.delay = float(delay)
        self.delay_steps = int(round(self.delay / self.dt))
        self.u_min = u_min
        self.u_max = u_max
        self._u_buffer: deque[float] = deque()

    # ------------------------------------------------------------- interface
    @property
    @abstractmethod
    def output(self) -> float:
        """Current plant output y."""

    @abstractmethod
    def _set_steady_state(self, y0: float) -> float:
        """Set internal states to rest at output y0, return the input holding it."""

    @abstractmethod
    def _integrate(self, u_delayed: float) -> None:
        """Advance internal states by one step dt."""

    # ---------------------------------------------------------------- common
    def saturate(self, u: float) -> float:
        if self.u_max is not None and u > self.u_max:
            return self.u_max
        if self.u_min is not None and u < self.u_min:
            return self.u_min
        return u

    def reset(self, y0: float = 0.0) -> float:
        """
        Put the plant at rest at output y0. The delay line is pre-filled with
        the steady-state input that holds the plant there (u(t < 0)).
        """
        u_steady = self._set_steady_state(float(y0))
        self._u_buffer = deque([u_steady] * self.delay_steps)
        return self.output

    def step(self, u: float) -> float:
        """Apply input u (saturated, then delayed) for one step; return new output."""
        self._u_buffer.append(self.saturate(float(u)))
        self._integrate(self._u_buffer.popleft())
        return self.output


class FirstOrderDelayPlant(Plant):
    """
    tau * dy/dt + y(t) = K * u(t - delay)

    :param tau: Time constant [s]
    :param gain: Static gain K
    """

    def __init__(self, tau: float = 1.0, gain: float = 1.0, delay: float = 0.5,
                 dt: float = 0.02, u_min: float | None = None, u_max: float | None = None):
        super().__init__(dt=dt, delay=delay, u_min=u_min, u_max=u_max)
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = float(tau)
        self.gain = float(gain)
        self.y = 0.0

    @property
    def output(self) -> float:
        return self.y

    def _set_steady_state(self, y0: float) -> float:
        self.y = y0
        return y0 / self.gain if self.gain != 0 else 0.0

    def _integrate(self, u_delayed: float) -> None:
        dydt = (-self.y + self.gain * u_delayed) / self.tau
        self.y = self.y + self.dt * dydt


class CascadeTanksPlant(Plant):
    """
    Two cascaded tanks ("object CF"):

        Ta * dy1/dt = K * u_real(t - delay) - 2 * sqrt(y_nom * y1)
        Tb * dy2/dt = y1 - y2,     y = y2

    Levels are physical, so y1, y2 >= 0 (a tank cannot hold a negative level).
    Steady state at output y: y1 = y2 = y, u = 2 * sqrt(y_nom * y) / K.

    :param ta: Time constant of the first (nonlinear) tank [s]
    :param tb: Time constant of the second tank [s]
    :param gain: Inflow gain K (equals the linearised static gain at y_nom)
    :param y_nom: Nominal operating point used in the outflow term
    :param u_min, u_max: Actuator saturation [U_min, U_max]
    """

    def __init__(self, ta: float = 1.0, tb: float = 0.5, gain: float = 1.0,
                 y_nom: float = 0.5, delay: float = 0.5, dt: float = 0.02,
                 u_min: float | None = 0.0, u_max: float | None = 2.0):
        super().__init__(dt=dt, delay=delay, u_min=u_min, u_max=u_max)
        if ta <= 0 or tb <= 0:
            raise ValueError("ta and tb must be positive")
        if y_nom <= 0:
            raise ValueError("y_nom must be positive")
        self.ta = float(ta)
        self.tb = float(tb)
        self.gain = float(gain)
        self.y_nom = float(y_nom)
        self.y1 = 0.0
        self.y2 = 0.0

    @property
    def output(self) -> float:
        return self.y2

    @property
    def linearised(self) -> tuple[float, float, float]:
        """(K, T2, T1) of G(s) = K / (T2 s^2 + T1 s + 1) around y_nom."""
        return self.gain, self.ta * self.tb, self.ta + self.tb

    def _outflow(self, y1: float) -> float:
        return 2.0 * math.sqrt(self.y_nom * max(y1, 0.0))

    def _set_steady_state(self, y0: float) -> float:
        if y0 < 0:
            raise ValueError("CascadeTanksPlant levels must be >= 0 (got y0 < 0)")
        self.y1 = self.y2 = y0
        return self._outflow(y0) / self.gain if self.gain != 0 else 0.0

    def _integrate(self, u_delayed: float) -> None:
        dy1 = (self.gain * u_delayed - self._outflow(self.y1)) / self.ta
        dy2 = (self.y1 - self.y2) / self.tb
        self.y1 = max(self.y1 + self.dt * dy1, 0.0)
        self.y2 = max(self.y2 + self.dt * dy2, 0.0)


# ----------------------------------------------------------------- registry

PLANT_MODELS = {
    "first_order": FirstOrderDelayPlant,
    "tanks": CascadeTanksPlant,
}


def make_plant(model: str, tau: float, dt: float, **params) -> Plant:
    """
    Build a plant for a scenario. The scenario's `tau` is the plant's main time
    constant: `tau` for the first-order plant, `ta` (first tank) for the tanks.
    """
    if model == "first_order":
        return FirstOrderDelayPlant(tau=tau, dt=dt, **params)
    if model == "tanks":
        return CascadeTanksPlant(ta=tau, dt=dt, **params)
    raise ValueError(f"Unknown plant model {model!r}; choose from {list(PLANT_MODELS)}")
