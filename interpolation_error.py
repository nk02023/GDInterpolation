import numpy as np
from scipy.misc import derivative

# Function to calculate the error bound for interpolation
def interpolation_error(f, x_values, x_eval):
    n = len(x_values) - 1
    
    # Function to compute the (n+1)th derivative of f
    def nth_derivative(f, x, n):
        return derivative(f, x, n=n, order=2*n+1)

    # Compute the error for each x in x_eval
    errors = []
    for x in x_eval:
        # Find the (n+1)th derivative at some point in the interval
        xi = np.random.uniform(min(x_values), max(x_values))  # Approximate ξ_x
        f_n1 = nth_derivative(f, xi, n+1)
        
        # Compute the product (x - x0)(x - x1)...(x - xn)
        product_term = np.prod([x - xj for xj in x_values])
        
        # Compute the error bound
        error = (f_n1 / np.math.factorial(n+1)) * product_term
        errors.append(error)
    
    return np.array(errors)

# Example usage
if __name__ == "__main__":
    # Define the function to interpolate
    def f(x):
        return np.sin(x)
    
    # Interpolation nodes
    x_values = np.array([0, 1, 2, 3])
    
    # Points to evaluate the error
    x_eval = np.linspace(0, 3, 100)
    
    # Compute the interpolation error
    errors = interpolation_error(f, x_values, x_eval)

    # Plotting the error
    import matplotlib.pyplot as plt
    plt.plot(x_eval, errors, label="Interpolation Error")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.title("Interpolation Error Using Theorem 2")
    plt.grid(True)
    plt.show()
# https://chatgpt.com/share/66fa0af8-7d70-800a-8e30-bd6eed2f90f6