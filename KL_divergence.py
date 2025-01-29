import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Create a random continuous probability distribution
# Using a mixture of normal distributions for variety
n_components = np.random.randint(1, 4)  # Random number of components (1-3)
weights = np.random.dirichlet(np.ones(n_components))  # Random weights that sum to 1
means = np.random.normal(0, 5, n_components)  # Random means
stds = np.random.uniform(0.5, 2, n_components)  # Random standard deviations

# Create the mixture distribution
P = lambda x: sum(w * stats.norm.pdf(x, loc=mu, scale=std) 
                 for w, mu, std in zip(weights, means, stds))

# Visualize the distribution
x = np.linspace(-10, 10, 1000)
y = [P(xi) for xi in x]

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='Distribution P')
plt.fill_between(x, y, alpha=0.3)
plt.title('Random Continuous Probability Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.legend()
plt.show()