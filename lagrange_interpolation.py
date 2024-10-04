import numpy as np
import matplotlib.pyplot as plt

# Function to compute the Lagrange basis polynomial L_i(x)
def lagrange_basis(x, i, x_points):
    basis = 1
    for j in range(len(x_points)):
        if j != i:
            basis *= (x - x_points[j]) / (x_points[i] - x_points[j])
    return basis

# Function to compute the Lagrange interpolating polynomial p(x)
def lagrange_interpolation(x, x_points, y_points):
    p = 0
    for i in range(len(x_points)):
        p += y_points[i] * lagrange_basis(x, i, x_points)
    return p

# Example points
x_points = np.array([0, 1, 2])
y_points = np.array([1, 3, 2])

# Generate a range of x values for plotting the polynomial
x_range = np.linspace(-1, 3, 100)
y_range = [lagrange_interpolation(x, x_points, y_points) for x in x_range]

# Plot the result
plt.plot(x_range, y_range, label="Lagrange Interpolant")
plt.scatter(x_points, y_points, color='red', label="Data Points")
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Lagrange Interpolating Polynomial')
plt.legend()
plt.grid(True)
plt.show()
