# Simulation using PID controller

import numpy as np
import matplotlib.pyplot as plt

from pid_controller import PIDController


def main():
    # Initialize the controller
    setpoint = 21   # Desired Temperature
    pid = PIDController(Kp=1.0, Ki=1.5, Kd=0.05, setpoint=setpoint)

    # Simulation parameters
    time = np.linspace(0, 10, 100)
    dt = time[1] - time[0]
    process_variable = 0
    process_values = []
    control_values = []

    # Simulate the process
    for t in time:
        # PID control output
        control_output = pid.compute(process_variable, dt)
        control_values.append(control_output)

        # Simulate process dynamics
        process_variable += control_output * dt - 0.1 * (process_variable - 20) * dt    # Heat loss

        # Store the process variable
        process_values.append(process_variable)

    # Show results
    plt.figure(figsize=(10,6))
    plt.title('PID Controller Simulation')
    plt.subplot(2,1,1)
    plt.plot(time, process_values, label='Process variable (Temperature)')
    plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [*C]')
    plt.legend()
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(time, control_values, label='Control output')
    plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
    plt.xlabel('Time [s]')
    plt.ylabel('Output value')
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == '__main__':
    main()