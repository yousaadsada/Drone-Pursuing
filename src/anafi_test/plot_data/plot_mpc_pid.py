import matplotlib.pyplot as plt
import pandas as pd

# Load your CSV file into a pandas DataFrame
data_mpc = pd.read_csv('/home/yousa/anafi_simulation/data/move2ref_point_mpc/drone_data.csv')
data_pid = pd.read_csv('/home/yousa/anafi_simulation/data/move2ref_point_pid/drone_data.csv')
data_pid_aggressive = pd.read_csv('/home/yousa/anafi_simulation/data/move2ref_point_pid_aggresive/drone_data.csv')

# Extract the relevant columns and convert them to numpy arrays
time = data_mpc['Timestamp'].to_numpy()
reference_x = data_mpc['Reference X'].to_numpy()
reference_y = data_mpc['Reference Y'].to_numpy()
reference_z = data_mpc['Reference Z'].to_numpy()
reference_yaw = data_mpc['Reference Yaw'].to_numpy()
current_x_mpc = data_mpc['Current X'].to_numpy()
current_y_mpc = data_mpc['Current Y'].to_numpy()
current_z_mpc = data_mpc['Current Z'].to_numpy()
current_yaw_mpc = data_mpc['Current Yaw'].to_numpy()

# Extract data for PID and PID aggressive
current_x_pid = data_pid['Current X'].to_numpy()
current_y_pid = data_pid['Current Y'].to_numpy()
current_z_pid = data_pid['Current Z'].to_numpy()
current_yaw_pid = data_pid['Current Yaw'].to_numpy()

current_x_pid_aggressive = data_pid_aggressive['Current X'].to_numpy()
current_y_pid_aggressive = data_pid_aggressive['Current Y'].to_numpy()
current_z_pid_aggressive = data_pid_aggressive['Current Z'].to_numpy()
current_yaw_pid_aggressive = data_pid_aggressive['Current Yaw'].to_numpy()

# Create four subplots, one for each figure
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot X vs time
axs[0, 0].plot(time, reference_x, label='Reference X', color='b')
axs[0, 0].plot(time, current_x_mpc, label='Current X MPC', color='g')
axs[0, 0].plot(time, current_x_pid, label='Current X PID Conservative', color='y')
axs[0, 0].plot(time, current_x_pid_aggressive, label='Current X PID Aggressive', color='r')
axs[0, 0].set_title('X vs Time')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('X (m)')
axs[0, 0].legend()

# Plot Y vs time
axs[0, 1].plot(time, reference_y, label='Reference Y', color='b')
axs[0, 1].plot(time, current_y_mpc, label='Current Y MPC', color='g')
axs[0, 1].plot(time, current_y_pid, label='Current Y PID Conservative', color='y')
axs[0, 1].plot(time, current_y_pid_aggressive, label='Current Y PID Aggressive', color='r')
axs[0, 1].set_title('Y vs Time')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Y (m)')
axs[0, 1].legend()

# Plot Z vs time
axs[1, 0].plot(time, reference_z, label='Reference Z', color='b')
axs[1, 0].plot(time, current_z_mpc, label='Current Z MPC', color='g')
axs[1, 0].plot(time, current_z_pid, label='Current Z PID Conservative', color='y')
axs[1, 0].plot(time, current_z_pid_aggressive, label='Current Z PID Aggressive', color='r')
axs[1, 0].set_title('Z vs Time')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Z (m)')
axs[1, 0].legend()

# Plot Yaw vs time
axs[1, 1].plot(time, reference_yaw, label='Reference Yaw', color='b')
axs[1, 1].plot(time, current_yaw_mpc, label='Current Yaw MPC', color='g')
axs[1, 1].plot(time, current_yaw_pid, label='Current Yaw PID Conservative', color='y')
axs[1, 1].plot(time, current_yaw_pid_aggressive, label='Current Yaw PID Aggressive', color='r')
axs[1, 1].set_title('Yaw vs Time')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Yaw (rad)')
axs[1, 1].legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()