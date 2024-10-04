# Theorem -3
# Tightness of the bound on the expected empirical loss
'''There is a data set f(xi; yi) 2 H × R : 1 ≤ i ≤ ng such that the mini-batch SGD
with update step (11) yields the following lower bound on the expected empirical quadratic loss
L(w)
E[L(wt)] = λ1 · Ehkδtk2i = λ1 · (g (m; η))t · Ehkδ0k2i'''
import numpy as np

# Generate data with feature vectors on the sphere and covariance matrix H with equal eigenvalues
def generate_spherical_data(n, d, beta):
    # Generate random feature vectors uniformly on a sphere of radius beta
    X = np.random.randn(n, d)
    X = beta * X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize to sphere of radius beta
    
    # True weights
    w_star = np.random.randn(d)
    
    # Generate labels with noise
    y = X.dot(w_star) + np.random.randn(n) * 0.1
    
    # Covariance matrix H with equal eigenvalues
    H = np.eye(d) * (beta**2 / d)
    
    # Eigenvalues are all the same
    eigvals = np.full(d, beta**2 / d)

    return X, y, w_star, H, eigvals

# SGD update step for quadratic loss
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

# Expected loss lower bound function
def expected_loss_lower_bound(eigvals, step_size, batch_size):
    def g_lambda(lam, step_size, m):
        return (1 - step_size * lam) ** 2

    g_values = [(g_lambda(lam, step_size, batch_size)) for lam in eigvals]
    
    return max(g_values)

# Run mini-batch SGD with empirical loss lower bounds
def run_sgd_with_loss_lower_bounds(X, y, w_init, H, eigvals, step_size, batch_size, max_iters):
    w = w_init
    losses = []
    loss_lower_bounds = []

    for t in range(max_iters):
        w = sgd_update(X, y, w, H, step_size, batch_size)
        
        # Quadratic loss
        loss = quadratic_loss(X, y, w)
        losses.append(loss)

        # Compute the loss lower bound according to the theorem
        g_m_eta = expected_loss_lower_bound(eigvals, step_size, batch_size)
        loss_lower_bound = eigvals[0] * (g_m_eta ** t)
        loss_lower_bounds.append(loss_lower_bound)

        if t % 100 == 0:
            print(f"Iteration {t}: Loss = {loss:.6f}, Loss lower bound = {loss_lower_bound:.6f}")

    return w, losses, loss_lower_bounds

# Quadratic loss function
def quadratic_loss(X, y, w):
    return (1 / len(y)) * np.sum((X.dot(w) - y) ** 2)

# Main function
if __name__ == "__main__":
    # Problem parameters
    n = 500   # Number of data points
    d = 100   # Dimensionality of the feature space
    beta = 10 # Radius of the sphere for the feature vectors
    step_size = 0.01  # Learning rate (η)
    batch_size = 32   # Mini-batch size (m)
    max_iters = 1000  # Number of iterations

    # Generate synthetic data where feature vectors lie on a sphere
    X, y, w_star, H, eigvals = generate_spherical_data(n, d, beta)

    # Initialize weights
    w_init = np.random.randn(d)

    # Run SGD with empirical loss lower bounds
    w_final, losses, loss_lower_bounds = run_sgd_with_loss_lower_bounds(X, y, w_init, H, eigvals, step_size, batch_size, max_iters)

    print("Final weights after SGD:", w_final)
    print("True weights:", w_star)
