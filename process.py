from collections import deque
import numpy as np

from parameters import K, TA, TB, Y0, TS, UMIN, UMAX_NOSAT

N_SUB = 10 #Euler steps per sampling period (TS/N_SUB)

class Process:
    def __init__(self, tau=0, umin=UMIN, umax=UMAX_NOSAT,
                 K=K, Ta=TA, Tb=TB, y0=Y0, Ts=TS):
        self.tau = int(tau)
        self.umin, self.umax = umin, umax
        self.K, self.Ta, self.Tb, self.y0, self.Ts = K, Ta, Tb, y0, Ts
        self.reset()

    def reset(self):
        self.y1 = 0.0
        self.y2 = 0.0
        self.buffer = deque([0.0] * self.tau, maxlen=self.tau or None)

    def step(self, u):
        """Apply control u for one sampling period, return output y."""
        u_real = min(max(u, self.umin), self.umax)  #saturation (before the delay)

        if self.tau > 0:  #transport delay
            u_del = self.buffer[0]
            self.buffer.append(u_real)
        else:
            u_del = u_real

        h = self.Ts / N_SUB #Euler integration with substeps
        for _ in range(N_SUB):
            dy1 = (self.K * u_del - 2.0 * np.sqrt(self.y0 * max(self.y1, 0.0))) / self.Ta
            dy2 = (self.y1 - self.y2) / self.Tb
            self.y1 += h * dy1
            self.y2 += h * dy2
        return self.y2