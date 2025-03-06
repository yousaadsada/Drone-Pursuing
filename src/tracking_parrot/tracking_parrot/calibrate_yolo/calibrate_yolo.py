import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

def load_csv(file_path):
    """Load the CSV file and extract timestamp, real, and estimated XYZ data."""
    df = pd.read_csv(file_path)

    # Extract data
    time_stamps = df['Timestamp'].values
    real_xyz = df[['Relative Pos Real X', 'Relative Pos Real Y', 'Relative Pos Real Z']].values
    estimated_xyz = df[['Relative Pos Yolo X', 'Relative Pos Yolo Y', 'Relative Pos Yolo Z']].values

    return time_stamps, real_xyz, estimated_xyz

def yaw_rotation_matrix(yaw):
    """Generate a 3x3 rotation matrix for a given yaw angle."""
    yaw = np.asarray(yaw).item()  # Extract scalar value if passed as an array
    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

def yaw_error_function(yaw, real_xyz, estimated_xyz):
    """Compute the RMSE for a given yaw rotation."""
    R_yaw = yaw_rotation_matrix(yaw)
    calibrated_xyz = estimated_xyz @ R_yaw.T  # Apply rotation
    error = np.linalg.norm(real_xyz - calibrated_xyz, axis=1)  # Euclidean distance per point
    return np.sqrt(np.mean(error**2))  # RMSE

def compute_optimal_yaw_rotation(real_xyz, estimated_xyz):
    """Find the optimal yaw angle that minimizes the RMSE."""
    result = minimize(yaw_error_function, x0=[0], args=(real_xyz, estimated_xyz), method='Powell')
    optimal_yaw = result.x[0]
    print(f"Optimal Yaw Angle: {np.degrees(optimal_yaw):.6f} degrees")
    return yaw_rotation_matrix(optimal_yaw)

def apply_rotation(R, estimated_xyz):
    """Apply the yaw-only rotation matrix R to estimated XYZ coordinates."""
    return estimated_xyz @ R.T  # Apply rotation

def compute_rmse(real_xyz, rotated_xyz):
    """Compute RMSE (Root Mean Square Error) between real and rotated XYZ."""
    error = np.linalg.norm(real_xyz - rotated_xyz, axis=1)  # Euclidean distance per point
    return np.sqrt(np.mean(error**2))

def save_rotation_matrix(R, output_file):
    """Save rotation matrix to a CSV file."""
    df_R = pd.DataFrame(R)
    df_R.to_csv(output_file, index=False, header=False)
    print(f"Yaw-Only Rotation matrix saved to {output_file}")

def plot_xyz(time_stamps, real_xyz, estimated_xyz, rotated_xyz, save_figure_dir):
    """Plot Real, Estimated, and Rotated XYZ values versus time."""
    labels = ['x', 'y', 'z']
    colors = ['r', 'g', 'b']
    for i in range(3):
        fig, axes = plt.subplots(1,1, figsize=(12, 8), sharex=True)
        axes.plot(time_stamps, real_xyz[:, i], color=colors[0], linestyle='-', label=f'Groud Truth {labels[i]}')
        axes.plot(time_stamps, estimated_xyz[:, i], color=colors[2], linestyle='--', label=f'YOLO Estimated {labels[i]}')
        
        axes.set_ylabel(f"{labels[i]} (m)", fontsize=18)  # Y-axis label larger
        axes.set_xlabel('Time (s)', fontsize=18)  # X-axis label larger
        axes.legend(loc="upper right", fontsize=16)  # Legend larger
        axes.tick_params(axis='both', labelsize=16)  # Tick labels larger


        save_path = os.path.join(save_figure_dir, f"{labels[i]}_vs_time.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')



    for i in range(3):
        fig, axes = plt.subplots(1,1, figsize=(12, 8), sharex=True)
        axes.plot(time_stamps, real_xyz[:, i], color=colors[0], linestyle='-', label=f'Groud Truth {labels[i]}')
        axes.plot(time_stamps, estimated_xyz[:, i], color=colors[2], linestyle='--', label=f'YOLO Estimated {labels[i]}')
        axes.plot(time_stamps, rotated_xyz[:, i], color=colors[1], linestyle='-.', label=f'YOLO Estimated with Calibration {labels[i]}')
        
        axes.set_ylabel(f"{labels[i]} (m)", fontsize=18)  # Y-axis label larger
        axes.set_xlabel('Time (s)', fontsize=18)  # X-axis label larger
        axes.legend(loc="upper right", fontsize=16)  # Legend larger
        axes.tick_params(axis='both', labelsize=16)  # Tick labels larger

        save_path = os.path.join(save_figure_dir, f"{labels[i]}_vs_time_calibrate.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    


if __name__ == "__main__":
    data_dir = "/home/yousa/anafi_simulation/data/tracking_parrot_yolo_test"
    file_path = os.path.join(data_dir, "drone_data.csv")  # Change this to your actual CSV file
    output_R_file = os.path.join(data_dir, "yaw_rotation_matrix.csv") 

    save_figure_dir = "/home/yousa/anafi_simulation/data/tracking_parrot_yolo_test/figure"
    os.makedirs(save_figure_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Load data
    time_stamps, real_xyz, estimated_xyz = load_csv(file_path)

    # Compute optimal yaw rotation matrix
    R_yaw = compute_optimal_yaw_rotation(real_xyz, estimated_xyz)

    # Save the rotation matrix
    save_rotation_matrix(R_yaw, output_R_file)

    # Apply yaw-only rotation to estimated data
    rotated_xyz = apply_rotation(R_yaw, estimated_xyz)

    # Compute RMSE
    rmse_value = compute_rmse(real_xyz, rotated_xyz)
    print(f"RMSE between real and yaw-calibrated XYZ: {rmse_value:.6f}")

    # Plot results
    plot_xyz(time_stamps, real_xyz, estimated_xyz, rotated_xyz, save_figure_dir)
