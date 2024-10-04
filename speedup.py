# The behavior of the speedup factor and critical batch size can be simulated in Python. 
# Below is a simple structure for calculating these quantities:
import numpy as np
import matplotlib.pyplot as plt

# Constants (assumed values, adjust based on the problem)
alpha = 0.1  # Strong convexity
beta = 1.0   # Smoothness constant
lambda_ = 0.5  # Regularization/curvature parameter
m_values = np.arange(1, 101)  # Batch sizes from 1 to 100

# Step size for batch size m
def eta_star(m, beta, lambda_):
    return (m * beta) / (beta + lambda_ * (m - 1))

# Speedup factor
def speedup_factor(m_values, beta, lambda_):
    eta_star_1 = eta_star(1, beta, lambda_)
    return eta_star(m_values, beta, lambda_) / eta_star_1

# Critical batch size
m_star = (beta / lambda_) + 1

# Calculate speedup factors for batch sizes
speedup_factors = speedup_factor(m_values, beta, lambda_)

# Plotting the speedup factor
plt.plot(m_values, speedup_factors, label='Speedup Factor')
plt.axvline(x=m_star, color='r', linestyle='--', label=f'Critical Batch Size: m* = {m_star:.2f}')
plt.title('Speedup Factor vs. Batch Size')
plt.xlabel('Batch Size (m)')
plt.ylabel('Speedup Factor (t(1) / t(m))')
plt.legend()
plt.grid(True)
plt.show()
