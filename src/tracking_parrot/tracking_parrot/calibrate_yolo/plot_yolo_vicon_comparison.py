import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
file_path = '/home/yousa/anafi_simulation/data/tracking_parrot_yolo_speed_prediction/drone_state_yolo_vicon_comparison.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)
data_dir = "/home/yousa/anafi_simulation/data/tracking_parrot_yolo_speed_prediction"

save_figure_dir = os.path.join(data_dir, "figures_moving")  
os.makedirs(save_figure_dir, exist_ok=True)

# Plot settings
plt.rcParams['figure.figsize'] = [10, 6]

# X Position Plot
plt.figure()
plt.plot(data['Timestamp'].values, data['Parrot X YOLO'].values, label='Parrot X YOLO', color = 'r')
plt.plot(data['Timestamp'].values, data['Parrot X Vicon'].values, label='Parrot X Vicon', color = 'g')
# plt.plot(data['Timestamp'].values, data['Parrot X RAW'].values, label='Parrot X RAW', color = 'b')
plt.title('X Position vs Time')
plt.xlabel('Time (s)')
plt.ylabel('X Position (m)')
plt.legend()
plt.grid()
save_path = os.path.join(save_figure_dir, 'x_position_vs_time.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Y Position Plot
plt.figure()
plt.plot(data['Timestamp'].values, data['Parrot Y YOLO'].values, label='Parrot Y YOLO', color = 'r')
plt.plot(data['Timestamp'].values, data['Parrot Y Vicon'].values, label='Parrot Y Vicon', color = 'g')
# plt.plot(data['Timestamp'].values, data['Parrot Y RAW'].values, label='Parrot Y RAW', color = 'b')
plt.title('Y Position vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.grid()
save_path = os.path.join(save_figure_dir, 'y_position_vs_time.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Z Position Plot
plt.figure()
plt.plot(data['Timestamp'].values, data['Parrot Z YOLO'].values, label='Parrot Z YOLO', color = 'r')
plt.plot(data['Timestamp'].values, data['Parrot Z Vicon'].values, label='Parrot Z Vicon',color = 'g')
# plt.plot(data['Timestamp'].values, data['Parrot Z RAW'].values, label='Parrot Z RAW', color = 'b')
plt.title('Z Position vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Z Position (m)')
plt.legend()
plt.grid()
save_path = os.path.join(save_figure_dir, 'z_position_vs_time.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# X Speed Plot
plt.figure()
plt.plot(data['Timestamp'].values, data['Parrot X_Speed YOLO'].values, label='Parrot X Speed YOLO', color = 'r')
plt.plot(data['Timestamp'].values, data['Parrot X_Speed Vicon'].values, label='Parrot X Speed Vicon', color = 'g')
# plt.plot(data['Timestamp'].values, data['Parrot X_Speed RAW'].values, label='Parrot X Speed Raw', color = 'b')
plt.title('X Speed vs Time')
plt.xlabel('Time (s)')
plt.ylabel('X Speed (m/s)')
plt.legend()
plt.grid()
save_path = os.path.join(save_figure_dir, 'x_speed_vs_time.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Y Speed Plot
plt.figure()
plt.plot(data['Timestamp'].values, data['Parrot Y_Speed YOLO'].values, label='Parrot Y Speed YOLO', color = 'r')
plt.plot(data['Timestamp'].values, data['Parrot Y_Speed Vicon'].values, label='Parrot Y Speed Vicon', color = 'g')
# plt.plot(data['Timestamp'].values, data['Parrot Y_Speed RAW'].values, label='Parrot Y Speed Raw', color = 'b')
plt.title('Y Speed vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Y Speed (m/s)')
plt.legend()
plt.grid()
save_path = os.path.join(save_figure_dir, 'y_speed_vs_time.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Z Speed Plot
plt.figure()
plt.plot(data['Timestamp'].values, data['Parrot Z_Speed YOLO'].values, label='Parrot Z Speed YOLO', color = 'r')
plt.plot(data['Timestamp'].values, data['Parrot Z_Speed Vicon'].values, label='Parrot Z Speed Vicon', color = 'g')
# plt.plot(data['Timestamp'].values, data['Parrot Z_Speed RAW'].values, label='Parrot Z Speed Raw', color = 'b')
plt.title('Z Speed vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Z Speed (m/s)')
plt.legend()
plt.grid()
save_path = os.path.join(save_figure_dir, 'z_speed_vs_time.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')



print("All plots saved in:", save_figure_dir)
