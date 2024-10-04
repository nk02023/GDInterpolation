import numpy as np
import sympy as sp

def lagrange_basis(x_values, x, j):
    """
    Lagrange basis polynomial L_j(x) = product of (x - x_m)/(x_j - x_m) for m != j.
    """
    basis_poly = 1
    for i in range(len(x_values)):
        if i != j:
            basis_poly *= (x - x_values[i]) / (x_values[j] - x_values[i])
    return basis_poly

def lagrange_interpolating_polynomial(x_values, y_values, x):
    """
    Calculate the Lagrange interpolating polynomial p_n(x).
    """
    interpolating_poly = 0
    for j in range(len(x_values)):
        interpolating_poly += y_values[j] * lagrange_basis(x_values, x, j)
    return interpolating_poly

def lagrange_error_bound(f, x_values, x):
    """
    Compute the error bound for equally spaced points.
    
    |E(x)| <= M / (n+1)! * product (x - xj) for j in [0, n]
    
    Here we approximate M as the maximum value of the (n+1)-th derivative of f on [a, b].
    """
    n = len(x_values) - 1
    x_symbol = sp.Symbol('x')
    
    # Compute the (n+1)-th derivative of f
    f_derivative = f.diff(x_symbol, n + 1)
    
    # Find the maximum of the (n+1)-th derivative on the interval [a, b]
    a, b = min(x_values), max(x_values)
    f_derivative_max = sp.lambdify(x_symbol, sp.Abs(f_derivative))
    
    # We approximate M by evaluating the derivative at several points in [a, b]
    x_eval_points = np.linspace(a, b, 1000)
    M = max(f_derivative_max(x_eval) for x_eval in x_eval_points)
    
    # Compute the product (x - xj) for j in [0, n]
    product_term = np.prod([(x - xj) for xj in x_values])
    
    # Calculate the error bound
    error_bound = M * abs(product_term) / sp.factorial(n + 1)
    
    return error_bound

if __name__ == "__main__":
    # Define the function f and the nodes
    x_symbol = sp.Symbol('x')
    f = sp.sin(x_symbol)  # Example function (sin(x))
    
    # Define equally spaced interpolation nodes and corresponding function values
    a, b = 0, np.pi  # Interval [a, b]
    n = 3  # Degree of the interpolating polynomial
    x_values = np.linspace(a, b, n+1)
    y_values = np.array([np.sin(x) for x in x_values])
    
    # Choose the point to evaluate the polynomial and error bound
    x_eval = np.pi / 4
    
    # Compute the interpolating polynomial
    poly_value = lagrange_interpolating_polynomial(x_values, y_values, x_eval)
    print(f"Interpolating polynomial at x = {x_eval}: {poly_value}")
    
    # Compute the general error bound
    error_bound_value = lagrange_error_bound(f, x_values, x_eval)
    print(f"General error bound at x = {x_eval}: {error_bound_value}")
