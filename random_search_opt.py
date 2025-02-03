import numpy as np
import matplotlib.pyplot as plt

# Define a sample objective function (2D function)
def sample_function(x):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.cos(3 * x[1]) * (1 - np.tanh(x[1] ** 2))

# Random search function (as defined earlier)
def Random_search(f, n_p, bounds_rs, iter_rs):
    localx   = np.zeros((n_p,iter_rs))  # Store sampled points
    localval = np.zeros((iter_rs))      # Store function values
    bounds_range = bounds_rs[:,1] - bounds_rs[:,0]
    bounds_bias  = bounds_rs[:,0]

    for sample_i in range(iter_rs):
        x_trial = np.random.uniform(0, 1, n_p)*bounds_range + bounds_bias  # Random sampling
        localx[:,sample_i] = x_trial
        localval[sample_i] = f(x_trial)  # Evaluate function

    minindex = np.argmin(localval)
    f_b      = localval[minindex]
    x_b      = localx[:,minindex]

    return f_b, x_b, localx, localval

# Define bounds for the search space
bounds_rs = np.array([[-2, 2], [-2, 2]])  # Search space for x0 and x1
iter_rs = 100  # Number of random samples
n_p = 2  # Two parameters

# Perform random search
best_f, best_x, sampled_x, sampled_f = Random_search(sample_function, n_p, bounds_rs, iter_rs)

# Plot results
fig, ax = plt.subplots(figsize=(8, 6))

# Generate a grid for visualization
x0_vals = np.linspace(bounds_rs[0, 0], bounds_rs[0, 1], 100)
x1_vals = np.linspace(bounds_rs[1, 0], bounds_rs[1, 1], 100)
X0, X1 = np.meshgrid(x0_vals, x1_vals)
Z = np.array([[sample_function([x0, x1]) for x0 in x0_vals] for x1 in x1_vals])

# Contour plot of the function
contour = ax.contourf(X0, X1, Z, levels=50, cmap="viridis")
plt.colorbar(contour, ax=ax, label="Function Value")

# Plot sampled points
ax.scatter(sampled_x[0, :], sampled_x[1, :], color='red', alpha=0.5, label="Sampled Points")

# Highlight the best point found
ax.scatter(best_x[0], best_x[1], color='white', edgecolors='black', s=100, label="Best Found")

ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_title("Random Search Optimization")
ax.legend()
plt.show()