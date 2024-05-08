"""
Authors: Abdelrahman Mohamed (22201965), Mazen Ghonem (00819365)

Description:
Custom Task 4.1: Multivariate Gaussian Distributions

This Python script generates synthetic data for two classes using multivariate Gaussian distributions
and plots scatter plots for three scenarios:
1. Both classes have the same covariance matrix that does not contain any 0.
2. Both classes have different covariance matrices.
3. Both classes have diagonal covariance matrices, but they are different.

Optional: It also plots the level contour of σ = 2 in the scatter plots.

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# Generate Synthetic Data
def generate_data(num_samples, mean, covariance):
    return np.random.multivariate_normal(mean, covariance, num_samples)


# Plot Scatter Plots
def plot_scatter(data1, data2, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data1[:, 0], data1[:, 1], label='Class 1')
    plt.scatter(data2[:, 0], data2[:, 1], label='Class 2')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


# Optional - Plot Level Contour
def plot_contour(covariance):
    plt.figure(figsize=(8, 6))
    x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    z = np.exp(-0.5 * (x**2 + y**2) / covariance)
    plt.contour(x, y, z, levels=[np.exp(-2)], colors='r')
    plt.title('Level Contour of σ = 2')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

# Define Parameters
num_samples = 1000
mean1 = np.array([1, 1])
mean2 = np.array([-1, -1])

# Generate Synthetic Data
covariance_same = np.array([[1, 0.5], [0.5, 1]])  # Same covariance matrix
covariance_diff = np.array([[1, 0.8], [0.8, 1.2]])  # Different covariance matrices
covariance_diag1 = np.array([[2, 0], [0, 1]])  # Diagonal covariance matrix for class 1
covariance_diag2 = np.array([[1, 0], [0, 2]])  # Diagonal covariance matrix for class 2

data1_same = generate_data(num_samples, mean1, covariance_same)
data2_same = generate_data(num_samples, mean2, covariance_same)

data1_diff = generate_data(num_samples, mean1, covariance_diff)
data2_diff = generate_data(num_samples, mean2, covariance_diff)

data1_diag = generate_data(num_samples, mean1, covariance_diag1)
data2_diag = generate_data(num_samples, mean2, covariance_diag2)

# Plot Scatter Plots
plot_scatter(data1_same, data2_same, 'Same Covariance Matrix')
plot_scatter(data1_diff, data2_diff, 'Different Covariance Matrices')
plot_scatter(data1_diag, data2_diag, 'Diagonal Covariance Matrices')

# Plot Level Contour
plot_contour(2)  # Plot level contour for σ = 2
