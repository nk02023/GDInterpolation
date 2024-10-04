# Theorem -2
# Upper bound on the expected empirical loss
import numpy as np
'''The following theorem provides an upper bound on the expected empirical loss after t iterations
of mini-batch SGD whose update step is given by '''
# Generate synthetic data and covariance matrix as in previous example
def generate_data(n, d, k):
    X = np.random.randn(n, d)
    w_star = np.random.randn(d)
    y = X.dot(w_star) + np.random.randn(n) * 0.1  # Add noise to the labels
    
    # Covariance matrix H with eigenvalue decomposition
    H = X.T.dot(X) / n
    eigvals, eigvecs = np.linalg.eigh(H)

    # Set the last d-k eigenvalues to 0
    eigvals[k:] = 0
    H = eigvecs.dot(np.diag(eigvals)).dot(eigvecs.T)

    return X, y, w_star, H, eigvals[:k]

# SGD update step (Theorem 2)
def sgd_update(X, y, w, H, step_size, batch_size):
    n, d = X.shape
    indices = np.random.choice(n, batch_size, replace=False)
    X_batch = X[indices]
    y_batch = y[indices]
    
    # Subsample covariance matrix
    H_batch = X_batch.T.dot(X_batch) / batch_size

    # Gradient of quadratic loss
    grad = H_batch.dot(w)
    
    # Update rule for SGD
    w_new = w - step_size * grad
    
    return w_new

# Expected loss bound function (Theorem 2)
def expected_loss_bound(eigvals, step_size, beta, batch_size):
    def g_lambda(lam, step_size, m, beta):
        return (1 - step_size * lam) ** 2 + (step_size ** 2) * lam * (beta - lam) / m

    g_values = [(g_lambda(lam, step_size, batch_size, beta)) for lam in eigvals]
    
    return max(g_values)

# Run mini-batch SGD with empirical loss bounds
def run_sgd_with_loss_bounds(X, y, w_init, H, eigvals, step_size, beta, batch_size, max_iters):
    w = w_init
    losses = []
    loss_bounds = []

    for t in range(max_iters):
        w = sgd_update(X, y, w, H, step_size, batch_size)
        
        # Quadratic loss
        loss = quadratic_loss(X, y, w)
        losses.append(loss)

        # Compute the loss bound according to the theorem
        g_m_eta = expected_loss_bound(eigvals, step_size, beta, batch_size)
        loss_bound = eigvals[0] * (g_m_eta ** t)
        loss_bounds.append(loss_bound)

        if t % 100 == 0:
            print(f"Iteration {t}: Loss = {loss:.6f}, Loss bound = {loss_bound:.6f}")

    return w, losses, loss_bounds

# Quadratic loss function
def quadratic_loss(X, y, w):
    return (1 / len(y)) * np.sum((X.dot(w) - y) ** 2)

# Main function
if __name__ == "__main__":
    # Problem parameters
    
    n = 500  # Number of data points
    d = 100  # Dimensionality of the feature space
    k = 50   # Rank of the covariance matrix H
    step_size = 0.01  # Learning rate (η)
    batch_size = 32   # Mini-batch size (m)
    max_iters = 1000  # Number of iterations
    beta = 1.0  # Upper bound for trace of covariance matrix

    # Generate synthetic data
    X, y, w_star, H, eigvals = generate_data(n, d, k)

    # Initialize weights
    w_init = np.random.randn(d)

    # Run SGD with empirical loss bounds
    w_final, losses, loss_bounds = run_sgd_with_loss_bounds(X, y, w_init, H, eigvals, step_size, beta, batch_size, max_iters)

    print("Final weights after SGD:", w_final)
    print("True weights:", w_star)
