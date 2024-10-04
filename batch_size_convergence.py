def optimal_step_size(m, beta, lambda1, lambdak):
    """
    Calculate the optimal step size η*(m) based on the given batch size m.
    
    Parameters:
    m (int): Batch size
    beta (float): A parameter
    lambda1 (float): Eigenvalue λ1
    lambdak (float): Eigenvalue λk
    
    Returns:
    float: Optimal step size η*(m)
    """
    if m <= beta:
        eta_star = m / (beta + (m - 1) * lambdak)
    else:
        eta_star = (lambda1 - lambdak + 1 / (2 * m)) / (beta + (m - 1) * (lambda1 + lambdak))
    
    return eta_star


def convergence_rate(m, beta, lambda1, lambdak):
    """
    Calculate the convergence rate g*(m) based on the given batch size m.
    
    Parameters:
    m (int): Batch size
    beta (float): A parameter
    lambda1 (float): Eigenvalue λ1
    lambdak (float): Eigenvalue λk
    
    Returns:
    float: Convergence rate g*(m)
    """
    if m <= beta:
        g_star = 1 - (m * lambdak) / (beta + (m - 1) * lambdak)
    else:
        g_star = 1 - (4 * m * (m - 1) * lambda1 * lambdak) / (beta + (m - 1) * (lambda1 + lambdak))**2
    
    return g_star


# Example usage
m_values = [1, 2, 3, 4, 5, 10, 15]
beta = 10
lambda1 = 1.5
lambdak = 0.5

for m in m_values:
    eta_star = optimal_step_size(m, beta, lambda1, lambdak)
    g_star = convergence_rate(m, beta, lambda1, lambdak)
    print(f"Batch Size: {m}, Optimal Step Size: {eta_star:.4f}, Convergence Rate: {g_star:.4f}")
