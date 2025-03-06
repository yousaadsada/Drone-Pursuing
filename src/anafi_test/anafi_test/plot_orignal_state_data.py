import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define paths to CSV files
csv_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data', 'real_drone_state_function', 'in_time_fourier_filter')
csv_file_x = os.path.join(csv_dir, 'state_data_1.csv')
csv_file_y = os.path.join(csv_dir, 'state_data_2.csv')
csv_file_z = os.path.join(csv_dir, 'state_data_3.csv')
csv_file_yaw = os.path.join(csv_dir, 'state_data_4.csv')

# Load data from CSV files
df_x = pd.read_csv(csv_file_x)
df_y = pd.read_csv(csv_file_y)
df_z = pd.read_csv(csv_file_z)
df_yaw = pd.read_csv(csv_file_yaw)


# Store data in a dictionary
data = {
    'x': df_x['x'].to_numpy(),
    'x_speed': df_x['x_speed'].to_numpy(),
    'y': df_y['y'].to_numpy(),
    'y_speed': df_y['y_speed'].to_numpy(),
    'z': df_z['z'].to_numpy(),
    'z_speed': df_z['z_speed'].to_numpy(),
    'yaw': df_yaw['pitch'].to_numpy(),
    'yaw_speed': df_yaw['pitch_speed'].to_numpy()
}

times = {
    'x': df_x['elapsed_time'].to_numpy(),
    'y': df_y['elapsed_time'].to_numpy(),
    'z': df_z['elapsed_time'].to_numpy(),
    'yaw': df_yaw['elapsed_time'].to_numpy(),
    'x_speed': df_x['elapsed_time'].to_numpy(),
    'y_speed': df_y['elapsed_time'].to_numpy(),
    'z_speed': df_z['elapsed_time'].to_numpy(),
    'yaw_speed': df_yaw['elapsed_time'].to_numpy(),

}


# Plotting
fig, axs = plt.subplots(4, 2, figsize=(12, 15))
keys = ['x', 'x_speed', 'y', 'y_speed', 'z', 'z_speed', 'yaw', 'yaw_speed']
titles = ['x_position', 'x_speed', 'y_position', 'y_speed', 'z_position', 'z_speed', 'yaw_position', 'yaw_speed']

# Loop to plot data
for index, axis in enumerate(keys):
    row = index // 2  # Determine row index
    col = index % 2   # Determine column index

    axs[row, col].plot(times[axis], data[axis], label=f'{axis.upper()}')
    axs[row, col].set_title(titles[index])
    axs[row, col].set_ylabel(f'{axis.upper()} Value')
    axs[row, col].legend()

        

plt.tight_layout()
plt.show()