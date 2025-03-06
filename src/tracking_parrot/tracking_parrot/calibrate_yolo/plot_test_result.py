import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_csv(file_path):
    """Load the CSV file and extract timestamp, real, estimated, and revised XYZ data."""
    df = pd.read_csv(file_path)

    # Extract columns
    time_stamps = df['Timestamp'].values
    real_xyz = df[['Relative Pos Real X', 'Relative Pos Real Y', 'Relative Pos Real Z']].values
    estimated_xyz = df[['Relative Pos Yolo X', 'Relative Pos Yolo Y', 'Relative Pos Yolo Z']].values
    revised_xyz = df[['Relative Pos Yolo Revise X', 'Relative Pos Yolo Revise Y', 'Relative Pos Yolo Revise Z']].values

    return time_stamps, real_xyz, estimated_xyz, revised_xyz

def plot_xyz(time_stamps, real_xyz, estimated_xyz, revised_xyz):
    """Plot Real, Estimated, and Revised XYZ values versus time."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']

    for i in range(3):
        axes[i].plot(time_stamps, real_xyz[:, i], color=colors[0], linestyle='-', label=f'Real {labels[i]}')
        axes[i].plot(time_stamps, estimated_xyz[:, i], color=colors[2], linestyle='-', label=f'Estimated {labels[i]}')
        axes[i].plot(time_stamps, revised_xyz[:, i], color=colors[1], linestyle='-', label=f'Revised {labels[i]}')
        
        axes[i].set_ylabel(labels[i])
        axes[i].legend()

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("Real vs Estimated vs Revised XYZ over Time")
    plt.show()

if __name__ == "__main__":
    # Change this to your actual CSV file path
    data_dir = "/home/yousa/anafi_simulation/data/tracking_parrot_yolo_test"
    file_path = os.path.join(data_dir, "drone_data_print.csv")

    # Load data
    time_stamps, real_xyz, estimated_xyz, revised_xyz = load_csv(file_path)

    # Plot results
    plot_xyz(time_stamps, real_xyz, estimated_xyz, revised_xyz)
