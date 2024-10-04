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

def lagrange_error(f, x_values, x):
    """
    Compute the Lagrange error E(x) using the formula:
    E(x) = f^{(n+1)}(ξx) / (n+1)! * product (x - xj) for j in [0, n]
    
    Here we approximate the (n+1)-th derivative at some ξx by using sympy.
    """
    n = len(x_values) - 1
    x_symbol = sp.Symbol('x')
    
    # Compute the (n+1)-th derivative of f
    f_derivative = f.diff(x_symbol, n + 1)
    
    # Compute the product (x - xj) for j in [0, n]
    product_term = np.prod([(x - xj) for xj in x_values])
    
    # Calculate the error term
    error = f_derivative.subs(x_symbol, x) * product_term / sp.factorial(n + 1)
    
    return error

if __name__ == "__main__":
    # Define the function f and the nodes
    x_symbol = sp.Symbol('x')
    f = sp.sin(x_symbol)  # Example function
    
    # Define the interpolation nodes and corresponding function values
    x_values = np.array([0, np.pi/2, np.pi])
    y_values = np.array([np.sin(x) for x in x_values])
    
    # Choose the point to evaluate the polynomial and error
    x_eval = np.pi / 4
    
    # Compute the interpolating polynomial
    poly_value = lagrange_interpolating_polynomial(x_values, y_values, x_eval)
    print(f"Interpolating polynomial at x = {x_eval}: {poly_value}")
    
    # Compute the error term
    error_value = lagrange_error(f, x_values, x_eval)
    print(f"Lagrange error at x = {x_eval}: {error_value}")
# https://chatgpt.com/share/66fa0af8-7d70-800a-8e30-bd6eed2f90f6