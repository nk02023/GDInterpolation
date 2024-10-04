import numpy as np

def convergence_rate_g_star(m, beta, lambda_k, lambda1):
    """
    Calculate the convergence rate g*(m) based on the batch size m.
    
    Parameters:
    m (int): Batch size
    beta (float): A parameter
    lambda_k (float): Eigenvalue λk
    lambda1 (float): Eigenvalue λ1
    
    Returns:
    float: Convergence rate g*(m)
    """
    if m <= beta:
        s_m = m / (1 + (m - 1) * lambda_k / beta)
    else:
        s_m = (4 * m * (m - 1) * lambda1) / (beta * (1 + (m - 1) * (lambda1 + lambda_k) / beta) ** 2)

    g_star = 1 - lambda_k / beta * s_m
    return g_star

def speed_up_factor_s(m, beta, lambda_k, lambda1):
    """
    Calculate the speed-up factor s(m) based on the batch size m.
    
    Parameters:
    m (int): Batch size
    beta (float): A parameter
    lambda_k (float): Eigenvalue λk
    lambda1 (float): Eigenvalue λ1
    
    Returns:
    float: Speed-up factor s(m)
    """
    if m <= beta:
        s_m = m / (1 + (m - 1) * lambda_k / beta)
    else:
        s_m = (4 * (m - 1) * lambda1) / (beta * (1 + (m - 1) * (lambda1 + lambda_k) / beta) ** 2)
    
    return s_m

# Example usage
beta = 0.5
lambda_k = 0.2
lambda1 = 1.5
m_values = range(1, 11)  # Batch sizes from 1 to 10

for m in m_values:
    g_star_value = convergence_rate_g_star(m, beta, lambda_k, lambda1)
    s_value = speed_up_factor_s(m, beta, lambda_k, lambda1)
    
    print(f"Batch Size: {m}, g*(m): {g_star_value:.4f}, s(m): {s_value:.4f}")

# Check for optimal batch size
optimal_m = 1
optimal_g_star = convergence_rate_g_star(optimal_m, beta, lambda_k, lambda1)
optimal_s = speed_up_factor_s(optimal_m, beta, lambda_k, lambda1)

print(f"\nOptimal Batch Size: {optimal_m}, g*(1): {optimal_g_star:.4f}, s(1): {optimal_s:.4f}")
