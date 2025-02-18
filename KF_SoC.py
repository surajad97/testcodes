import numpy as np
import matplotlib.pyplot as plt

# Simulated battery parameters
capacity_Ah = 2.0  # Battery capacity in Ah
R_internal = 0.02  # Internal resistance in Ohms
OCV_SoC_curve = lambda soc: 3.0 + 0.8 * soc  # Approximate OCV-SoC relationship

def simulate_battery_voltage(soc, current):
    """Simulates measured battery voltage given SoC and current draw."""
    return OCV_SoC_curve(soc) - current * R_internal

# Kalman Filter Parameters
A = 1  # State transition (SoC evolution)
B = -1 / (capacity_Ah * 3600)  # Current impact on SoC
C = 1  # Observation model (voltage)
Q = 1e-6  # Process noise covariance
R = 1e-3  # Measurement noise covariance
P = 0.1  # Initial estimate uncertainty

# Initial state estimate (assume 50% SoC)
x_hat = 0.46

# Simulated test data
num_steps = 100

time = np.linspace(0, 100, num_steps)
currents = np.sin(time / 10) * 0.5  # Simulated current draw (A)
true_soc = np.zeros(num_steps)
measured_voltages = np.zeros(num_steps)
estimated_soc = np.zeros(num_steps)

for k in range(num_steps):
    # True SoC (using simple Coulomb counting)
    if k > 0:
        true_soc[k] = true_soc[k-1] + B * currents[k]
    else:
        true_soc[k] = x_hat
    
    # Simulate measured voltage
    measured_voltages[k] = simulate_battery_voltage(true_soc[k], currents[k]) + np.random.normal(0, np.sqrt(R))
    
    # Kalman Filter - Prediction Step
    x_hat_minus = A * x_hat + B * currents[k]
    P_minus = A * P * A + Q
    
    # Kalman Gain
    K = P_minus * C / (C * P_minus * C + R)
    
    # Kalman Filter - Update Step
    x_hat = x_hat_minus + K * (measured_voltages[k] - OCV_SoC_curve(x_hat_minus))
    P = (1 - K * C) * P_minus
    
    # Store estimated SoC
    estimated_soc[k] = x_hat

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(time, true_soc, label="True SoC", linestyle='dashed')
plt.plot(time, estimated_soc, label="Estimated SoC (Kalman)", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("SoC (%)")
plt.legend()
plt.title("Kalman Filter for Battery SoC Estimation")
plt.grid()
plt.show()
