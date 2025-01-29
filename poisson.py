import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Set lambda parameter
lambda_param = 18

# Generate x values (0 to 15 should cover most of the distribution)
x = np.arange(0, 60)

# Calculate Poisson PMF
pmf = poisson.pmf(x, lambda_param)

# Create the plot
plt.figure(figsize=(10, 6))
plt.bar(x, pmf, alpha=0.8, color='green', label=f'λ = {lambda_param}')
plt.title('Poisson Distribution')
plt.xlabel('k (number of events)')
plt.ylabel('Probability')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
