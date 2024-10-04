import numpy as np

# Function to calculate the divided differences table
def divided_differences(x_values, y_values):
    n = len(x_values)
    dd_table = np.zeros((n, n))
    dd_table[:, 0] = y_values  # The first column is y-values

    # Fill the divided difference table
    for j in range(1, n):
        for i in range(n - j):
            dd_table[i, j] = (dd_table[i+1, j-1] - dd_table[i, j-1]) / (x_values[i+j] - x_values[i])
    
    return dd_table

# Function to evaluate the Newton polynomial at a given point x
def newton_polynomial(x_values, dd_table, x):
    n = len(x_values)
    result = dd_table[0, 0]  # Start with the first divided difference
    product_term = 1.0

    # Add the terms of the Newton polynomial
    for j in range(1, n):
        product_term *= (x - x_values[j-1])  # (x - x0)(x - x1)...(x - xj-1)
        result += dd_table[0, j] * product_term

    return result

# Main function to interpolate and evaluate
def newton_interpolation(x_values, y_values, x_eval):
    # Step 1: Calculate divided differences
    dd_table = divided_differences(x_values, y_values)
    
    # Step 2: Evaluate the Newton polynomial at each point in x_eval
    y_eval = [newton_polynomial(x_values, dd_table, x) for x in x_eval]
    
    return np.array(y_eval)

# Example usage
if __name__ == "__main__":
    # Interpolation points
    x_values = np.array([1, 2, 4, 7])
    y_values = np.array([3, 6, 8, 10])

    # Points to evaluate the Newton polynomial
    x_eval = np.linspace(1, 7, 100)

    # Get the interpolated values
    y_eval = newton_interpolation(x_values, y_values, x_eval)

    # Plotting the result
    import matplotlib.pyplot as plt
    plt.plot(x_eval, y_eval, label="Newton Interpolation", color="red")
    plt.scatter(x_values, y_values, label="Data points", color="blue")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Newton's Interpolating Polynomial")
    plt.grid(True)
    plt.show()
