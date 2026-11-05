import numpy as np
from scipy.optimize import minimize

from pid import PID
from process import Process
from sim import run_episode
from zn import zn_from_step

T_END = 40.0        
MAX_GAIN = 15.0         
BIG = 1e3               
START_SCALES = (0.3, 1.0)   
STEP = 0.4               


def evaluate(gains, tau, r, umax, umin=0.0, t_end=T_END):
    pid = PID(*gains, umin=umin, umax=umax)
    plant = Process(tau=tau, umin=umin, umax=umax)
    return run_episode(pid, plant, r, t_end=t_end)


def cost(logg, tau, r, umax, umin=0.0):
    gains = np.exp(logg)
    if np.any(gains > MAX_GAIN):
        return BIG
    res = evaluate(gains, tau, r, umax, umin)
    y = res["y"]
    if not np.all(np.isfinite(y)) or y.max() > 10 * r:
        return BIG
    return res["itae"]


def optimize_gains(tau, r, umax, umin=0.0):
    z = zn_from_step()
    zn_log = np.log([z["Kp"], z["Ki"], z["Kd"]])

    best = None
    for scale in START_SCALES:
        x0 = zn_log + np.log(scale)
        simplex = np.vstack([x0, x0 + STEP * np.eye(3)])
        sol = minimize(cost, x0, args=(tau, r, umax, umin), method="Nelder-Mead",
                       options={"initial_simplex": simplex, "xatol": 1e-3,
                                "fatol": 1e-4, "maxiter": 600})
        if best is None or sol.fun < best.fun:
            best = sol

    Kp, Ki, Kd = np.exp(best.x)
    return Kp, Ki, Kd