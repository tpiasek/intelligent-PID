import numpy as np

from parameters import K, TA, TB


def step_response(t, K=K, Ta=TA, Tb=TB):
    return K * (1.0 - (Ta * np.exp(-t / Ta) - Tb * np.exp(-t / Tb)) / (Ta - Tb))

def zn_from_step(K=K, Ta=TA, Tb=TB, t_end=60.0, dt=0.001):
    t = np.arange(0.0, t_end, dt)
    y = step_response(t, K, Ta, Tb)

    slope = np.gradient(y, dt)
    i = int(np.argmax(slope))              
    t_i, y_i, s_i = t[i], y[i], slope[i]

    t_zero = t_i - y_i / s_i               
    t_gain = t_i + (K - y_i) / s_i          
    L = t_zero
    Tz = t_gain - t_zero

    Kp = 1.2 * Tz / (K * L)
    Ti = 2.0 * L
    Td = 0.5 * L
    return {"L": L, "Tz": Tz, "Kp": Kp, "Ki": Kp / Ti, "Kd": Kp * Td}


if __name__ == "__main__":
    z = zn_from_step()
    print(f"L = {z['L']:.3f} s, Tz = {z['Tz']:.3f} s")
    print(f"Kp = {z['Kp']:.3f}, Ki = {z['Ki']:.3f}, Kd = {z['Kd']:.3f}")