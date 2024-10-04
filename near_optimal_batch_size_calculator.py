def convergence_rate_g_hat(m, beta, lambda_k, lambda1):
    """
    Calculate the near-optimal convergence rate g^(m) based on the batch size m.
    
    Parameters:
    m (int): Batch size
    beta (float): A parameter
    lambda_k (float): Eigenvalue λk
    lambda1 (float): Eigenvalue λ1
    
    Returns:
    float: Near-optimal convergence rate g^(m)
    """
    if m <= beta:
        g_hat = 1 - m * lambda_k / (beta * (1 + (m - 1) / n))
    else:
        g_hat = (lambda1 - lambda_k + 1) * (1 - 4 * m * (m - 1) * lambda1 * lambda_k / 
                                             (beta + (m - 1) * (lambda1 + lambda_k))**2)

    return g_hat

# Example usage
beta = 0.5
lambda_k = 0.2
lambda1 = 1.5
n = 100  # Total number of samples
m_values = range(1, 11)  # Batch sizes from 1 to 10

for m in m_values:
    g_hat_value = convergence_rate_g_hat(m, beta, lambda_k, lambda1)
    
    print(f"Batch Size: {m}, g^(m): {g_hat_value:.4f}")

# Check for optimal batch size
optimal_m_hat = 1
optimal_g_hat = convergence_rate_g_hat(optimal_m_hat, beta, lambda_k, lambda1)

print(f"\nOptimal Batch Size: {optimal_m_hat}, g^(1): {optimal_g_hat:.4f}")
