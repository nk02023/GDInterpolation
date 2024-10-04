import numpy as np

# Define the degree of the polynomial
n = 3  # Degree of the polynomial

# Define two polynomials p(x) and q(x)
# Coefficients are listed from highest degree to lowest (np.poly1d does this by default)
p_coeff = [1, -4, 6, -3]  # Polynomial p(x) = x^3 - 4x^2 + 6x - 3
q_coeff = [1, -4, 6, -3]  # Polynomial q(x) = x^3 - 4x^2 + 6x - 3 (same as p)

p = np.poly1d(p_coeff)
q = np.poly1d(q_coeff)

# Define n+1 distinct points
points = np.linspace(-1, 1, n+1)  # Generates n+1 distinct points in the range [-1, 1]

# Evaluate both polynomials at the n+1 distinct points
p_values = p(points)
q_values = q(points)

# Check if p(x) and q(x) are equal at all n+1 points
if np.allclose(p_values, q_values):
    print("p(x) and q(x) are equal at all n+1 points, so p(x) = q(x).")
else:
    print("p(x) and q(x) are not equal at some points.")

# Verify that r(x) = p(x) - q(x) is the zero polynomial
r = np.poly1d(np.array(p_coeff) - np.array(q_coeff))

print("r(x) =", r)
