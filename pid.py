from parameters import TS, UMIN, UMAX_NOSAT

class PID:

    def __init__(self, Kp, Ki, Kd, Ts=TS, umin=UMIN, umax=UMAX_NOSAT):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.Ts = Ts
        self.umin, self.umax = umin, umax
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_e = 0.0

    def step(self, r, y):
        """Compute control u from setpoint r and measured output y."""
        e = r - y

        d = (e - self.prev_e) / self.Ts
        self.prev_e = e

        integral_new = self.integral + e * self.Ts
        u = self.Kp * e + self.Ki * integral_new + self.Kd * d

        # anti-windup: accept the new integral only if u is within limits
        if self.umin <= u <= self.umax:
            self.integral = integral_new
        else:
            u = self.Kp * e + self.Ki * self.integral + self.Kd * d

        return min(max(u, self.umin), self.umax)