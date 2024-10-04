def step_size_eta_hat(m, beta, lambda1, n):
    """
    Calculate the step size η^(m) based on the given batch size m.
    
    Parameters:
    m (int): Batch size
    beta (float): A parameter
    lambda1 (float): Eigenvalue λ1
    n (float): Total number of samples
    
    Returns:
    float: Step size η^(m)
    """
    if m <= beta:
        eta_hat = m / (beta * (1 + (m - 1) / n))
    else:
        eta_hat = (lambda1 - beta / n + 1 / (2 * m)) / (beta + (m - 1) * (lambda1 + beta / n))
    
    return eta_hat


def upper_bound_convergence_rate(m, beta, lambda1, n):
    """
    Calculate the upper bound g^(m) on the convergence rate based on the given batch size m.
    
    Parameters:
    m (int): Batch size
    beta (float): A parameter
    lambda1 (float): Eigenvalue λ1
    n (float): Total number of samples
    
    Returns:
    float: Upper bound convergence rate g^(m)
    """
    if m <= beta:
        g_hat = 1 - (m * beta) / (beta * (1 + (m - 1) / n))
    else:
        g_hat = 1 - (4 * m * (m - 1) * lambda1 * beta) / (beta + (m - 1) * (lambda1 + beta / n))**2
    
    return g_hat


# Example usage
m_values = [1, 2, 3, 4, 5, 10, 15]
beta = 0.5
lambda1 = 1.5
n = 20

for m in m_values:
    eta_hat = step_size_eta_hat(m, beta, lambda1, n)
    g_hat = upper_bound_convergence_rate(m, beta, lambda1, n)
    print(f"Batch Size: {m}, Step Size η^: {eta_hat:.4f}, Upper Bound g^: {g_hat:.4f}")
