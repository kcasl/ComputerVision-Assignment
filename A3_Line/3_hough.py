'''
    Prof. Seokju Lee @ EL4001
    Python code for Hough Transform
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb


# (1) Generate random 2D points near two random lines
def generate_random_points_near_two_lines(num_points, x_range=(0, 10), noise_level=0.2):
    # Split the total number of points into two sets, one for each line
    num_points_line1 = num_points // 2
    num_points_line2 = num_points - num_points_line1  # Remaining points for the second line

    # Define two random lines in the form of y = m1*x + b1 and y = m2*x + b2
    m1 = np.random.uniform(-1, 1)  # Random slope for the first line
    b1 = np.random.uniform(-5, 5)  # Random y-intercept for the first line
    m2 = np.random.uniform(-1, 1)  # Random slope for the second line
    b2 = np.random.uniform(-5, 5)  # Random y-intercept for the second line

    # Generate random x values for the points on the two lines
    x1 = np.random.uniform(x_range[0], x_range[1], num_points_line1)
    x2 = np.random.uniform(x_range[0], x_range[1], num_points_line2)

    # Calculate corresponding y values for the two lines and add noise
    y1 = m1 * x1 + b1 + np.random.normal(0, noise_level, num_points_line1)
    y2 = m2 * x2 + b2 + np.random.normal(0, noise_level, num_points_line2)

    # Combine the points from both lines
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    return x, y, m1, b1, m2, b2


# (2) Hough Line Transform (Direct Implementation with line merging and accumulator return)
def hough_line_transform(x, y, num_rho=360, num_theta=180, threshold=25, min_distance=10, min_angle=np.deg2rad(5)):
    # Define the ranges for rho and theta
    max_rho = int(np.hypot(max(x), max(y)))  # Maximum rho is the diagonal of the image
    rhos = np.linspace(-max_rho, max_rho, num_rho)  # Possible values of rho
    thetas = np.linspace(-np.pi / 2, np.pi / 2, num_theta)  # Theta from -90 to 90 degrees

    # Accumulator array to store votes
    accumulator = np.zeros((len(rhos), len(thetas)))

    # Hough transform: for each point, calculate rho and theta and increment accumulator
    for xi, yi in zip(x, y):
        for theta_idx, theta in enumerate(thetas):
            rho = xi * np.cos(theta) + yi * np.sin(theta)
            rho_idx = np.argmin(np.abs(rhos - rho))  # Find the nearest rho index
            accumulator[rho_idx, theta_idx] += 1

    # Bug fix on the accumulator map
    accumulator[:,0] = 0
    accumulator[:,-1] = 0
    accumulator[0,:] = 0
    accumulator[-1,:] = 0

    # Extract the lines that exceed the threshold
    lines = []
    for rho_idx in range(len(rhos)):
        for theta_idx in range(len(thetas)):
            if accumulator[rho_idx, theta_idx] >= threshold:
                rho = rhos[rho_idx]
                theta = thetas[theta_idx]

                # Check for similar lines and merge them
                too_similar = False
                for rho_existing, theta_existing in lines:
                    if (np.abs(rho - rho_existing) < min_distance) and (np.abs(theta - theta_existing) < min_angle):
                        too_similar = True
                        break

                if not too_similar:
                    lines.append((rho, theta))

    return lines, accumulator, rhos, thetas


# (3) Plot the random points, the true line, the Hough detected lines, and the accumulator map
def plot_points_and_hough_lines(x, y, m1_true, b1_true, m2_true, b2_true, lines, accumulator, rhos, thetas):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the random points and true lines
    axs[0].scatter(x, y, label="Random Points")
    x_line = np.linspace(min(x), max(x), 100)
    y_true = m1_true * x_line + b1_true
    axs[0].plot(x_line, y_true, 'b--', label=f"True Line 1: y={m1_true:.2f}x + {b1_true:.2f}")
    y_true = m2_true * x_line + b2_true
    axs[0].plot(x_line, y_true, 'g--', label=f"True Line 2: y={m2_true:.2f}x + {b2_true:.2f}")

    # Plot the detected lines
    for rho, theta in lines:
        if np.sin(theta) != 0:
            x_hough = np.linspace(min(x), max(x), 100)
            y_hough = (rho - x_hough * np.cos(theta)) / np.sin(theta)
            axs[0].plot(x_hough, y_hough, 'r-', label=f"Hough Line: rho={rho:.2f}, theta={theta:.2f}")

    axs[0].set_xlim([0, max(x) + 1])
    axs[0].set_ylim([min(y) - 1, max(y) + 1])
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title('Hough Line Transform Detection')
    axs[0].legend()
    axs[0].grid(True)

    # Plot the accumulator map
    cax = axs[1].imshow(accumulator, extent=[thetas[0], thetas[-1], rhos[0], rhos[-1]], aspect='auto', cmap='hot')
    axs[1].set_xlabel('Theta (radians)')
    axs[1].set_ylabel('Rho')
    axs[1].set_title('Accumulator Map')
    fig.colorbar(cax, ax=axs[1])

    plt.tight_layout(); plt.ion(); plt.show()


# (4) Main execution code
if __name__ == "__main__":
    # Generate 200 random points near two random lines
    num_points = 200
    x, y, m1_true, b1_true, m2_true, b2_true = generate_random_points_near_two_lines(num_points)

    # Perform Hough Line Transform on the points
    lines, accumulator, rhos, thetas = hough_line_transform(x, y, num_rho=360, num_theta=360, threshold=16, \
                                                            min_distance=10, min_angle=np.deg2rad(5))

    # Print the detected lines
    if lines:
        print(f"Detected {len(lines)} lines using Hough Transform.")
        for rho, theta in lines:
            print(f"Line detected: rho={rho:.2f}, theta={theta:.2f}")
    else:
        print("No lines detected.")

    # Plot the result with Hough lines and accumulator map
    plot_points_and_hough_lines(x, y, m1_true, b1_true, m2_true, b2_true, lines, accumulator, rhos, thetas)
    # pdb.set_trace()
