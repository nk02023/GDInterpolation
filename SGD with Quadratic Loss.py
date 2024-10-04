import numpy as np

# Generate synthetic data for the quadratic loss problem
def generate_data(n, d, k):
    """
    Generate synthetic data for the problem where:
    n - number of samples
    d - dimensionality of feature space
    k - number of non-zero eigenvalues (rank of the covariance matrix H)
    """
    # Generate random feature matrix X
    X = np.random.randn(n, d)
    
    # Generate true weights w_star such that L(w*) = 0
    w_star = np.random.randn(d)

    # Generate labels
    y = X.dot(w_star) + np.random.randn(n) * 0.1  # add some noise

    # Covariance matrix H with rank k
    H = X.T.dot(X) / n
    eigvals, eigvecs = np.linalg.eigh(H)

    # Set the last d-k eigenvalues to 0
    eigvals[k:] = 0
    H = eigvecs.dot(np.diag(eigvals)).dot(eigvecs.T)

    return X, y, w_star, H

# Quadratic loss function
def quadratic_loss(X, y, w):
    """
    Calculate the quadratic loss L(w) = (1/n) * ||Xw - y||^2
    """
    return (1 / len(y)) * np.sum((X.dot(w) - y) ** 2)

# Stochastic Gradient Descent (SGD) update
def sgd_update(X, y, w, H, step_size, batch_size):
    """
    Perform one SGD update with mini-batch size and step size.
    X - data matrix (n, d)
    y - labels
    w - current weights
    H - covariance matrix
    step_size - learning rate
    batch_size - size of the mini-batch
    """
    n, d = X.shape
    indices = np.random.choice(n, batch_size, replace=False)
    X_batch = X[indices]
    y_batch = y[indices]
    
    # Subsample covariance matrix
    H_batch = X_batch.T.dot(X_batch) / batch_size

    # Gradient of quadratic loss
    grad = H_batch.dot(w - w_star)

    # Update rule for SGD
    w_new = w - step_size * grad

    return w_new

# SGD for minimizing quadratic loss
def sgd_quadratic_loss(X, y, w_init, H, step_size, batch_size, max_iters):
    """
    Perform SGD to minimize the quadratic loss.
    X - data matrix (n, d)
    y - labels
    w_init - initial weights
    H - covariance matrix
    step_size - learning rate
    batch_size - size of the mini-batch
    max_iters - maximum number of iterations
    """
    w = w_init
    losses = []

    for i in range(max_iters):
        w = sgd_update(X, y, w, H, step_size, batch_size)
        loss = quadratic_loss(X, y, w)
        losses.append(loss)

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss:.6f}")

    return w, losses

# Main function to run the program
if __name__ == "__main__":
    # Parameters
    n = 500  # number of data points
    d = 100  # dimensionality of the feature space
    k = 50   # rank of the covariance matrix H
    step_size = 0.01  # learning rate
    batch_size = 32   # mini-batch size
    max_iters = 1000  # number of iterations

    # Generate data
    X, y, w_star, H = generate_data(n, d, k)

    # Initialize weights
    w_init = np.random.randn(d)

    # Perform SGD
    w_final, losses = sgd_quadratic_loss(X, y, w_init, H, step_size, batch_size, max_iters)

    print("Final weights after SGD:", w_final)
    print("True weights:", w_star)
