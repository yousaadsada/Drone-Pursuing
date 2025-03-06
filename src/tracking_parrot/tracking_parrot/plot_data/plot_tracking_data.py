import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
file_path = "/home/yousa/anafi_simulation/data/tracking_parrot_yolo/drone_data.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

fig = plt.figure(figsize=(10,6))

ax3d = fig.add_subplot(projection = '3d')
ax3d.view_init(elev=20, azim=150)
 
ax3d.set_title("Real-Time 3D Trajectory")
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.set_xlim([-2, 2])
ax3d.set_ylim([-2, 2])
ax3d.set_zlim([0, 3])

time_stamp = df["Timestamp"].to_numpy()[:500]
parrot_x = df["Parrot X"].to_numpy()[:500]
parrot_y = df["Parrot Y"].to_numpy()[:500]
parrot_z = df["Parrot Z"].to_numpy()[:500]
anafi_x = df["Anafi X"].to_numpy()[:500]
anafi_y = df["Anafi Y"].to_numpy()[:500]
anafi_z = df["Anafi Z"].to_numpy()[:500]

ax3d.set_title("Real-Time 3D Trajectory")
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.set_xlim([-2, 2])
ax3d.set_ylim([-2, 2])
ax3d.set_zlim([0, 3])

ax3d.plot(parrot_x, parrot_y, parrot_z, 'ro-', label="Parrot")
ax3d.plot(anafi_x, anafi_y, anafi_z, 'bo-', label="Anafi")

ax3d.legend()

plt.savefig("/home/yousa/anafi_simulation/data/tracking_parrot_yolo/3d_trajectory_plot.png", dpi=300, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(10,8))
ax_x = fig.add_subplot(3, 1, 1)
ax_y = fig.add_subplot(3, 1, 2)
ax_z = fig.add_subplot(3, 1, 3)

ax_x.set_title("X vs Time")
ax_y.set_title("Y vs Time")
ax_z.set_title("Z vs Time")

ax_x.set_ylabel("X Position")
ax_y.set_ylabel("Y Position")
ax_z.set_ylabel("Z Position")

ax_x.set_xlabel("Time (s)")
ax_y.set_xlabel("Time (s)")
ax_z.set_xlabel("Time (s)")

ax_x.set_xlim([0, 20])
ax_y.set_xlim([0, 20])
ax_z.set_xlim([0, 20])

ax_x.set_ylim([-3, 3.5])
ax_y.set_ylim([-2, 2])
ax_z.set_ylim([0.5, 2])

ax_x.plot(time_stamp, parrot_x, 'r-', label="Parrot X")
ax_y.plot(time_stamp, parrot_y, 'g-', label="Parrot Y")
ax_z.plot(time_stamp, parrot_z, 'b-', label="Parrot Z")

ax_x.plot(time_stamp, anafi_x, 'r--', label="Anafi X")
ax_y.plot(time_stamp, anafi_y, 'g--', label="Anafi Y")
ax_z.plot(time_stamp, anafi_z, 'b--', label="Anafi Z")

ax_x.legend(loc="upper right", fontsize=10, frameon=True)
ax_y.legend(loc="upper right", fontsize=10, frameon=True)
ax_z.legend(loc="upper right", fontsize=10, frameon=True)

plt.subplots_adjust(hspace=0.8) 
plt.savefig("/home/yousa/anafi_simulation/data/tracking_parrot_yolo/2d_trajectory_plot.png", dpi=300, bbox_inches="tight")
plt.show()
