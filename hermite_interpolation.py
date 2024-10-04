import numpy as np

# Function to compute the Hermite divided differences
def hermite_divided_diff(x, f, f_prime):
    n = len(x)
    Q = np.zeros((2*n, 2*n))  # Matrix for divided differences
    z = np.zeros(2*n)  # Array for repeated nodes

    # Fill in the z array (repeated x-values) and initial divided difference
    for i in range(n):
        z[2*i] = z[2*i+1] = x[i]
        Q[2*i, 0] = Q[2*i+1, 0] = f[i]
        Q[2*i+1, 1] = f_prime[i]
        if i != 0:
            Q[2*i, 1] = (Q[2*i, 0] - Q[2*i-1, 0]) / (z[2*i] - z[2*i-1])

    # Fill in the rest of the divided difference table
    for i in range(2, 2*n):
        for j in range(2, i+1):
            Q[i, j] = (Q[i, j-1] - Q[i-1, j-1]) / (z[i] - z[i-j])

    return Q, z

# Function to compute Hermite interpolation polynomial
def hermite_polynomial(x_data, f_data, f_prime_data, x):
    Q, z = hermite_divided_diff(x_data, f_data, f_prime_data)
    n = len(x_data)

    # Start with the first term of the divided difference table
    result = Q[0, 0]

    # Compute the polynomial iteratively
    product_term = 1
    for i in range(1, 2*n):
        product_term *= (x - z[i-1])
        result += Q[i, i] * product_term

    return result

# Reading the data from the file
def read_data_from_file(filename):
    x_data = []
    f_data = []
    f_prime_data = []
    
    with open(filename, 'r') as file:
        for line in file:
            values = line.split()
            x_data.append(float(values[0]))
            f_data.append(float(values[1]))
            f_prime_data.append(float(values[2]))
    
    return x_data, f_data, f_prime_data

# Main function to test Hermite interpolation
if __name__ == "__main__":
    # Read data from file
    filename = "hermite_data.txt"
    x_data, f_data, f_prime_data = read_data_from_file(filename)

    # Test the Hermite interpolant
    x_test = 0.5  # Point where we want to interpolate
    result = hermite_polynomial(x_data, f_data, f_prime_data, x_test)
    
    print(f"Hermite interpolant at x = {x_test} is: {result}")

# https://chatgpt.com/share/66fa0af8-7d70-800a-8e30-bd6eed2f90f6