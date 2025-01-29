import numpy as np
import matplotlib.pyplot as plt

# Time vector
t = np.linspace(0, 1, 1000)  # 1 second duration, 1000 samples

# Generate low frequency signal (2 Hz sine wave)
low_freq = 2
signal = 3 * np.sin(2 * np.pi * low_freq * t)

# Add high frequency noise (20 Hz sine wave + random noise)
high_freq = 20
noise = np.sin(2 * np.pi * high_freq * t) + 0.5 * np.random.randn(len(t))
noisy_signal = signal + noise




# Plot the signals
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, signal, 'b-', label='Clean Signal (2 Hz)')
plt.title('Original Low Frequency Signal')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, noise, 'b-', label='High Frequency Noise')
plt.title('High Frequency Noise')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, noisy_signal, 'r-', label='Signal + Noise')
plt.title('Noisy Signal (2 Hz + 20 Hz + Random Noise)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
