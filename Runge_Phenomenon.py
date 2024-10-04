import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

# Define the Runge function
def runge_function(x):
    return 1 / (1 + 25 * x**2)

# Interpolation points
def equally_spaced_points(a, b, n):
    return np.linspace(a, b, n+1)

# Interpolate and plot the results
def interpolate_and_plot_runge(n, ax):
    # Equally spaced points
    x_values = equally_spaced_points(-1, 1, n)
    y_values = runge_function(x_values)
    
    # Perform Lagrange interpolation
    poly = lagrange(x_values, y_values)
    
    # Evaluate the interpolating polynomial on a dense grid
    x_dense = np.linspace(-1, 1, 1000)
    y_dense = runge_function(x_dense)
    y_interp = poly(x_dense)
    
    # Plot the original function and the interpolation
    ax.plot(x_dense, y_dense, 'b-', label="Runge function", linewidth=2)
    ax.plot(x_dense, y_interp, 'r--', label=f"Lagrange polynomial (n={n})", linewidth=2)
    ax.scatter(x_values, y_values, color='red', zorder=5)
    ax.set_title(f"Runge's Phenomenon (n={n})")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()

if __name__ == "__main__":
    # Create subplots to show different degrees of interpolation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Interpolate and plot the Runge function for different degrees
    interpolate_and_plot_runge(5, axes[0])   # Polynomial of degree 5
    interpolate_and_plot_runge(10, axes[1])  # Polynomial of degree 10
    interpolate_and_plot_runge(15, axes[2])  # Polynomial of degree 15
    
    plt.tight_layout()
    plt.show()
