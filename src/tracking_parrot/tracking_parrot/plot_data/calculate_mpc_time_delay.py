import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

data_dir = "/home/yousa/anafi_simulation/data/mpc_delay" 
axes = ['x', 'y', 'z']
speed_levels = [1,2,3,4,5,6,7,8,9,10]
save_dir = "/home/yousa/anafi_simulation/data/mpc_delay/result" 

os.makedirs(save_dir, exist_ok=True)

for axis in axes:

    mpc_time_delay = []

    for speed in speed_levels:
        csv_file = os.path.join(data_dir, f'state_data_{axis}_{speed}.csv')

        if not os.path.exists(csv_file):
            print(f"File not founding: {csv_file}")
            continue

        df = pd.read_csv(csv_file)

        time_stamp = df["time_stamp"].to_numpy()
        ref = df[f"ref_{axis}"].to_numpy()
        drone = df[f"drone_{axis}"].to_numpy()

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title(f"{axis.upper()} Speed:{speed} (m/s)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")

        ax.plot(time_stamp, ref, 'r-', label=f"Ref {axis.upper()}")
        ax.plot(time_stamp, drone, 'g-', label=f"Drone {axis.upper()}")

        ax.legend(loc="upper right", fontsize=10, frameon=True)

        save_path = os.path.join(save_dir, f"{axis}_{speed}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        ref_calculation = df[f"ref_{axis}"].to_numpy()[50:]
        drone_calculation = df[f"drone_{axis}"].to_numpy()[50:]

        difference = ref_calculation - drone_calculation
        average_difference = np.mean(difference)

        mpc_time_delay.append((average_difference / int(speed)*10))
    
    mpc_time_delay_csv_file = os.path.join(save_dir, f"mpc_time_delay_{axis}.csv")
    df_delay = pd.DataFrame(mpc_time_delay, columns=["MPC_Time_Delay"])
    df_delay.to_csv(mpc_time_delay_csv_file, index=False)

    mpc_time_delay_average_csv_file = os.path.join(save_dir, f"mpc_time_delay_average_{axis}.csv")
    average_mpc_time_delay = sum(mpc_time_delay) / len(mpc_time_delay) if mpc_time_delay else 0
    df_delay_average = pd.DataFrame([[average_mpc_time_delay]], columns=["MPC_Time_Delay_Average"])
    df_delay_average.to_csv(mpc_time_delay_average_csv_file, index=False)








        
