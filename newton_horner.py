from newton_poly_interpolation import divided_diff
def newton_horner(x, x_points, coef):
    """
    Evaluates the Newton interpolating polynomial at x using Horner's method.
    
    Parameters:
        x : float
            The point at which to evaluate the polynomial.
        x_points : list or array
            The abscissas (x0, x1, ..., xn).
        coef : list or array
            The divided difference coefficients (c0, c1, ..., cn).
    
    Returns:
        float
            The value of the Newton polynomial at x.
    """
    n = len(coef) - 1
    y = coef[n]
    
    # Apply Horner's method
    for k in range(n - 1, -1, -1):
        y = coef[k] + (x - x_points[k]) * y
    
    return y

# Example points
x_points = [0, 1, 2]
y_points = [1, 3, 2]

# Get the divided differences coefficients
coef = divided_diff(x_points, y_points)

# Evaluate the Newton polynomial using Horner's method
x_eval = 1.5
y_eval = newton_horner(x_eval, x_points, coef)

print(f"The Newton interpolating polynomial evaluated at x = {x_eval} is {y_eval:.4f}")


# https://chatgpt.com/share/66f98e8c-6200-800a-846b-c33144f717c5