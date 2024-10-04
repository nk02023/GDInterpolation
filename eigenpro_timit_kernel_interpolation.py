import numpy as np
import timit
from sklearn.kernel_approximation import RBFSampler
from eigenpro import EigenPro

# Load TIMIT dataset
def load_timit_data():
    # Simplified loading, assumes you have the dataset locally
    timit_dataset = timit.load_data()
    
    # Extract features and labels
    # For the sake of simplicity, let's assume timit_dataset provides numpy arrays
    X_train, y_train = timit_dataset.train_features, timit_dataset.train_labels
    X_test, y_test = timit_dataset.test_features, timit_dataset.test_labels
    
    return X_train, y_train, X_test, y_test

# Define RBF kernel approximation
def rbf_kernel_approximation(X_train, X_test, gamma=0.1, n_components=100):
    rbf_feature = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)
    X_train_rbf = rbf_feature.fit_transform(X_train)
    X_test_rbf = rbf_feature.transform(X_test)
    
    return X_train_rbf, X_test_rbf

# Train with EigenPro optimizer
def eigenpro_training(X_train, y_train, X_test, y_test, batch_size=256, n_iter=10):
    # Normalize the labels
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # EigenPro training
    model = EigenPro(batch_size=batch_size, n_epochs=n_iter)
    
    # Fit model
    model.fit(X_train, y_train)

    # Evaluate on test set
    test_accuracy = model.score(X_test, y_test)
    
    print(f'Test accuracy: {test_accuracy:.4f}')
    return model

if __name__ == "__main__":
    # Load the TIMIT dataset
    X_train, y_train, X_test, y_test = load_timit_data()
    
    # Interpolate using RBF kernel approximation
    X_train_rbf, X_test_rbf = rbf_kernel_approximation(X_train, X_test)

    # Train using EigenPro
    model = eigenpro_training(X_train_rbf, y_train, X_test_rbf, y_test)
