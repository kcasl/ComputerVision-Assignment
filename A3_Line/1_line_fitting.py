'''
    Prof. Seokju Lee @ EL4001
    Python code for line fitting
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import pdb


# (1) Generate random 2D points near a random line
def generate_random_points_near_line(num_points, x_range=(0, 10), noise_level=0.3):
    # Define a random line in the form of y = mx + b
    m = np.random.uniform(-1, 1)  # Random slope
    b = np.random.uniform(-5, 5)  # Random y-intercept

    # Generate random x values
    x = np.random.uniform(x_range[0], x_range[1], num_points)

    # Calculate corresponding y values for the line and add noise
    y = m * x + b + np.random.normal(0, noise_level, num_points)

    return x, y, m, b


# (2) Function to minimize the sum of squared perpendicular distances
def total_perpendicular_distance(params, x, y):
    # params: [theta, rho]
    theta, rho = params

    # pdb.set_trace()
    
    ################################
    ### YOUR IMPLEMENTATION HERE ###
    dist = x * np.cos(theta) + y * np.sin(theta) - rho
    return np.sum(dist ** 2)
    ################################


# (3) Fit a line using minimize function
def fit_line_optimize(x, y):
    # Initial guess: theta = 0, rho = 0
    initial_guess = [0, 0]
    
    # Use minimize function to optimize theta and rho, on the target function: total_perpendicular_distance
    result = minimize(total_perpendicular_distance, initial_guess, args=(x, y))     
    
    # Return optimized theta and rho
    theta_opt, rho_opt = result.x
    return theta_opt, rho_opt


# (4) Plot the random points and fitted line
def plot_points_and_fitted_line(x, y, theta, rho, m_true, b_true):
    plt.scatter(x, y, label="Random Points")
    
    # Plot the fitted line (convert from polar coordinates to Cartesian)
    x_line = np.linspace(min(x), max(x), 100)
    y_line = (rho - x_line * np.cos(theta)) / np.sin(theta)
    
    # Plot the true random line (for comparison)
    y_true = m_true * x_line + b_true
    plt.plot(x_line, y_true, 'b--', label=f"True Line: y={m_true:.2f}x + {b_true:.2f}")
    plt.plot(x_line, y_line, color='r', label=f"Fitted Line: theta={theta:.2f}, rho={rho:.2f}")
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Line Fitting using Perpendicular Distance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout(); plt.ion(); plt.show()


# (5) Main execution code
if __name__ == "__main__":
    # Generate 50 random points near a random line
    num_points = 50
    x, y, m_true, b_true = generate_random_points_near_line(num_points)
    
    # Fit a line in polar coordinates
    theta_opt, rho_opt = fit_line_optimize(x, y)
    
    print(f"Optimal theta: {theta_opt:.2f}, Optimal rho: {rho_opt:.2f}")
    
    # Plot the result
    plot_points_and_fitted_line(x, y, theta_opt, rho_opt, m_true, b_true)
