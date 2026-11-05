import numpy as np

from parameters import TS

def run_episode(pid, plant, r, t_end=60.0, ts=TS, band=0.02):
    pid.reset()
    plant.reset()
    n = int(round(t_end / ts))
    t = np.arange(1, n + 1) * ts
    y_arr = np.zeros(n)
    u_arr = np.zeros(n)

    y = 0.0
    for k in range(n):
        u = pid.step(r, y)
        y = plant.step(u)
        y_arr[k], u_arr[k] = y, u

    e = r - y_arr
    itae = float(np.sum(t * np.abs(e)) * ts)
    overshoot = float(max(0.0, (y_arr.max() - r) / r * 100.0))

    outside = np.where(np.abs(e) > band * r)[0]
    if len(outside) == 0:
        settling = 0.0
    elif outside[-1] == n - 1:
        settling = float("inf")       
    else:
        settling = float(t[outside[-1] + 1])

    return {"t": t, "y": y_arr, "u": u_arr,
            "itae": itae, "overshoot": overshoot, "settling": settling}