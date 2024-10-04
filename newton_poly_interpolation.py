import numpy as np
import matplotlib.pyplot as plt

# Function to compute divided differences
def divided_diff(x_points, y_points):
    n = len(y_points)
    coef = np.zeros([n, n])
    coef[:,0] = y_points  # First column is y values
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x_points[i+j] - x_points[i])
    
    return coef[0]  # Return the first row (coefficients for the Newton polynomial)

# Function to compute Newton polynomial
def newton_poly(coef, x_points, x):
    n = len(coef) - 1
    p = coef[n]
    for k in range(1, n+1):
        p = coef[n-k] + (x - x_points[n-k]) * p
    return p

# Example points
x_points = np.array([0, 1, 2])
y_points = np.array([1, 3, 2])

# Get the divided differences coefficients
coef = divided_diff(x_points, y_points)

# Generate a range of x values for plotting the polynomial
x_range = np.linspace(-1, 3, 100)
y_range = [newton_poly(coef, x_points, x) for x in x_range]

# Plot the result
plt.plot(x_range, y_range, label="Newton Interpolant")
plt.scatter(x_points, y_points, color='red', label="Data Points")
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Newton Interpolating Polynomial')
plt.legend()
plt.grid(True)
plt.show()
