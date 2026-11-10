'''
    Prof. Seokju Lee @ EL4001
    Python code for RANSAC
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb


# (1) Generate random 2D points near a random line
def generate_random_points_near_line(num_points, x_range=(0, 10), noise_level=0.3, outlier_ratio=0.2):
    # Define a random line in the form of y = mx + b
    m = np.random.uniform(-1, 1)  # Random slope
    b = np.random.uniform(-5, 5)  # Random y-intercept

    # Generate random x values
    x = np.random.uniform(x_range[0], x_range[1], num_points)

    # Calculate corresponding y values for the line and add noise
    y = m * x + b + np.random.normal(0, noise_level, num_points)

    # Add outliers
    num_outliers = int(outlier_ratio * num_points)
    x[:num_outliers] = np.random.uniform(x_range[0], x_range[1], num_outliers)
    y[:num_outliers] = np.random.uniform(min(y), max(y), num_outliers)

    return x, y, m, b


# (2) Define the model for line fitting (y = mx + b)
def fit_line_model(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b


# (3) RANSAC algorithm for line fitting
def ransac(x, y, num_iterations=1000, threshold=1.0, min_inliers=20):
    best_m, best_b = None, None
    best_inlier_count = 0
    
    for _ in range(num_iterations):
        # Randomly sample 2 points (minimum)
        sample_indices = np.random.choice(len(x), 2, replace=False)
        x_sample, y_sample = x[sample_indices], y[sample_indices]
        
        # Fit a line to these 2 points
        m, b = fit_line_model(x_sample, y_sample)
        
        # Calculate distances from all points to the line
        # distances = np.abs(y - (m * x + b))  # Vertical distance to the line
        distances = np.abs(m * x - y + b) / np.sqrt(m**2 + 1)  # Perpendicular distance to the line

        # Calculate inliers
        inliers = distances < threshold
        
        # If more inliers than the current best model, update!
        ################################
        ### YOUR IMPLEMENTATION HERE ###
        inlier_count = np.sum(inliers)
        if inlier_count > best_inlier_count and inlier_count >= min_inliers:
            best_inlier_count = inlier_count
            best_m, best_b = m, b
        ################################
    
    return best_m, best_b


# (4) Plot the random points and the RANSAC fitted line
def plot_points_and_ransac_line(x, y, m_fitted, b_fitted, m_true, b_true):
    plt.scatter(x, y, label="Random Points")
    
    # Plot the fitted line
    x_line = np.linspace(min(x), max(x), 100)
    y_fitted = m_fitted * x_line + b_fitted
    
    # Plot the true random line (for comparison)
    y_true = m_true * x_line + b_true
    plt.plot(x_line, y_true, 'b--', label=f"True Line: y={m_true:.2f}x + {b_true:.2f}")
    plt.plot(x_line, y_fitted, color='r', label=f"RANSAC Fitted Line: y={m_fitted:.2f}x + {b_fitted:.2f}")
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Line Fitting using RANSAC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout(); plt.ion(); plt.show()


# (5) Main execution code
if __name__ == "__main__":
    # Generate 50 random points near a random line, with some outliers
    num_points = 50
    x, y, m_true, b_true = generate_random_points_near_line(num_points)
    
    # Fit a line using RANSAC
    m_ransac, b_ransac = ransac(x, y)
    
    print(f"RANSAC Fitted Line: y = {m_ransac:.2f}x + {b_ransac:.2f}")
    
    # Plot the result
    plot_points_and_ransac_line(x, y, m_ransac, b_ransac, m_true, b_true)
    # pdb.set_trace()
