# Theorem -1
# The theorem below shows exponential convergence for mini-batch SGD in the interpolated regime.
''' Theorem 1. For the setting described above and under Assumption 1, for any mini-batch size
 m 2 N, the SGD iteration (1) with constant step size η∗(m) , m
 β+λ(m−1) gives the following guarantee
'''

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for linear regression
np.random.seed(42)
n_samples = 100
X = 2 * np.random.rand(n_samples, 1)
y = 4 + 3 * X + np.random.randn(n_samples, 1)

# Add bias term to X
X_b = np.c_[np.ones((n_samples, 1)), X]

# Initialize parameters
w = np.random.randn(2, 1)

# Parameters for SGD and the theorem
alpha = 0.01  # Strong convexity parameter
beta = 0.1    # Smoothness parameter
lambda_ = 0.1 # Regularization parameter
iterations = 1000
m = 10        # Mini-batch size
eta_star = m / (beta + lambda_ * (m - 1))  # Optimal step size

# Loss function (mean squared error)
def compute_loss(X_b, y, w):
    m = len(y)
    predictions = X_b.dot(w)
    loss = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return loss

# Gradient of the loss function
def compute_gradient(X_b, y, w):
    m = len(y)
    predictions = X_b.dot(w)
    gradients = (1 / m) * X_b.T.dot(predictions - y)
    return gradients

# Stochastic Gradient Descent with theorem-based error bounds
error_history = []
loss_history = []

for iteration in range(iterations):
    mini_batch_indices = np.random.randint(0, n_samples, m)
    X_b_mini_batch = X_b[mini_batch_indices]
    y_mini_batch = y[mini_batch_indices]
    
    gradient = compute_gradient(X_b_mini_batch, y_mini_batch, w)
    
    # Update weights
    w = w - eta_star * gradient
    
    # Compute the current loss
    loss = compute_loss(X_b, y, w)
    loss_history.append(loss)
    
    # Simulate the error bound (distance to optimal solution, assuming w* is zero)
    delta_t = np.linalg.norm(w - np.zeros_like(w))
    error_history.append(delta_t**2)
    
    # Stop if the loss is small enough
    if loss < 0.01:
        break

# Plotting the loss function over time
plt.plot(loss_history, label='Loss')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("SGD Loss Convergence")
plt.legend()
plt.show()

# Plotting the error bound (theoretical convergence bound)
plt.plot(error_history, label='Error Bound (Distance to Optimal)')
plt.xlabel("Iterations")
plt.ylabel("Squared Error (||w_t - w*||^2)")
plt.title("Error Bound Convergence")
plt.legend()
plt.show()

print("Final parameters (weights):", w)
