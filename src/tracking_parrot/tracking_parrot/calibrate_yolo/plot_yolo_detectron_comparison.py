import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_csv(file_path, output_file_path):
    """Load CSV file and extract timestamp, ground truth, YOLO calibration, and Detectron calibration data."""
    df = pd.read_csv(file_path)

    # Extract columns
    time_stamps = df['Timestamp'].values
    ground_truth_xyz = df[['Relative Pos Ground Truth X', 'Relative Pos Ground Truth Y', 'Relative Pos Ground Truth Z']].values
    yolo_calibrated_xyz = df[['Relative Pos Yolo Calibration X', 'Relative Pos Yolo Calibration Y', 'Relative Pos Yolo Calibration Z']].values
    detectron_calibrated_xyz = df[['Relative Pos Detectron Calibration X', 'Relative Pos Detectron Calibration Y', 'Relative Pos Detectron Calibration Z']].values


    ref = np.mean(ground_truth_xyz, axis=0)

# Calculate bias
    yolo_bias = np.mean(yolo_calibrated_xyz - ref, axis=0)
    detectron_bias = np.mean(detectron_calibrated_xyz - ref, axis=0)

    # Calculate noise
    yolo_noise = np.std(yolo_calibrated_xyz - (ref + yolo_bias), axis=0)
    detectron_noise = np.std(detectron_calibrated_xyz - (ref + detectron_bias), axis=0)

    # Print the results
    print("Reference (Ground Truth Average):", ref)
    print("\nYOLO Calibration Bias:", yolo_bias)
    print("YOLO Calibration Noise:", yolo_noise)

    print("\nDetectron Calibration Bias:", detectron_bias)
    print("Detectron Calibration Noise:", detectron_noise)


    output_data = {
    'Metric': ['Reference_X', 'Reference_Y', 'Reference_Z',
               'YOLO_Bias_X', 'YOLO_Bias_Y', 'YOLO_Bias_Z',
               'Detectron_Bias_X', 'Detectron_Bias_Y', 'Detectron_Bias_Z',
               'YOLO_Noise_X', 'YOLO_Noise_Y', 'YOLO_Noise_Z',
               'Detectron_Noise_X', 'Detectron_Noise_Y', 'Detectron_Noise_Z'],
    'Value': np.concatenate([ref, yolo_bias, detectron_bias, yolo_noise, detectron_noise])
    }

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file_path, index=False)

    # Identify zero values for each dataset separately
    zero_mask_ground_truth = np.any(ground_truth_xyz == 0.0, axis=1)
    zero_mask_yolo = np.any(yolo_calibrated_xyz == 0.0, axis=1)
    zero_mask_detectron = np.any(detectron_calibrated_xyz == 0.0, axis=1)

    # Replace zero rows with NaN (this creates separate discontinuities for each dataset)
    ground_truth_xyz[zero_mask_ground_truth] = np.nan
    yolo_calibrated_xyz[zero_mask_yolo] = np.nan
    detectron_calibrated_xyz[zero_mask_detectron] = np.nan

    return time_stamps, ground_truth_xyz, yolo_calibrated_xyz, detectron_calibrated_xyz

def plot_xyz(time_stamps, ground_truth_xyz, yolo_calibrated_xyz, detectron_calibrated_xyz, save_figure_dir):
    """Plot XYZ data versus TimeStamp with separate discontinuities for each dataset."""
    
    labels = ['X', 'Y', 'Z']
    colors = {'ground_truth': 'r', 'yolo': 'g', 'detectron': 'b'}
    
    # Ensure save directory exists
    os.makedirs(save_figure_dir, exist_ok=True)

    for i in range(3):
        fig, axes = plt.subplots(1, 1, figsize=(12, 8), sharex=True)

        # Plot each dataset separately with its own discontinuities
        axes.plot(time_stamps, ground_truth_xyz[:, i], color=colors['ground_truth'], linestyle='-', label=f'Ground Truth {labels[i]}')
        axes.plot(time_stamps, yolo_calibrated_xyz[:, i], color=colors['yolo'], linestyle='-', label=f'YOLO Calibration {labels[i]}')
        axes.plot(time_stamps, detectron_calibrated_xyz[:, i], color=colors['detectron'], linestyle='-', label=f'Detectron Calibration {labels[i]}')

        # Set labels with larger font sizes
        axes.set_ylabel(f"{labels[i]} (m)", fontsize=18)
        axes.set_xlabel('Time (s)', fontsize=18)
        axes.legend(loc="upper right", fontsize=16)
        axes.tick_params(axis='both', labelsize=16)

        # Save figure
        save_path = os.path.join(save_figure_dir, f"{labels[i]}_vs_time.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"Figures saved in: {save_figure_dir}")

if __name__ == "__main__":
    # Define file paths
    data_dir = "/home/yousa/anafi_simulation/data/tracking_parrot_yolo_test"
    # file_path = os.path.join(data_dir, "drone_data_yolo_detectron_comparison_moving.csv")
    # save_figure_dir = os.path.join(data_dir, "figures_moving")

    file_path = os.path.join(data_dir, "drone_data_yolo_detectron_comparison.csv")
    output_file_path = os.path.join(data_dir, "noise_bias_yolo_detectron.csv")
    save_figure_dir = os.path.join(data_dir, "figures")

    # Load data
    time_stamps, ground_truth_xyz, yolo_calibrated_xyz, detectron_calibrated_xyz = load_csv(file_path, output_file_path)

    # Plot and save figures
    plot_xyz(time_stamps, ground_truth_xyz, yolo_calibrated_xyz, detectron_calibrated_xyz, save_figure_dir)
