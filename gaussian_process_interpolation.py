import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Sample training data
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2.5, 3.5, 2.0, 3.0, 4.0])

# Define the kernel
kernel = RBF(length_scale=1.0)

# Create Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel)

# Fit the model to training data
gp.fit(X_train, y_train)

# Make predictions
X_test = np.array([[1.5], [2.5], [3.5], [4.5]])
y_pred, y_std = gp.predict(X_test, return_std=True)

print("Predicted values:", y_pred)
print("Standard deviations:", y_std)
