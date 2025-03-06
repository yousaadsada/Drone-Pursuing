import rclpy
from rclpy.node import Node
import matplotlib
matplotlib.use("TkAgg")  # Ensure Matplotlib runs properly
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from anafi_msg.msg import PlotData
import threading
from rclpy.executors import MultiThreadedExecutor

class RealTimePlot(Node):
    def __init__(self):
        super().__init__('real_time_plot')

        # Data storage
        self.time_data = []
        self.x_parrot, self.y_parrot, self.z_parrot = [], [], []
        self.x_anafi, self.y_anafi, self.z_anafi = [], [], []

        # ROS2 Subscriber
        self.sub_data = self.create_subscription(
            PlotData, '/plotdata', self.get_data_callback, 10
        )

        self.start_plot_threads()

    def get_data_callback(self, msg):
        """ ROS2 Subscriber callback - stores received data. """
        self.get_logger().info(f"Received Data - x_parrot: {msg.x_parrot}")

        # Store data
        current_time = msg.time_stamp
        self.time_data.append(current_time)
        self.x_parrot.append(msg.x_parrot)
        self.y_parrot.append(msg.y_parrot)
        self.z_parrot.append(msg.z_parrot)
        self.x_anafi.append(msg.x_anafi)
        self.y_anafi.append(msg.y_anafi)
        self.z_anafi.append(msg.z_anafi)

    def start_plot_threads(self):
        """ Start two separate Matplotlib windows in different threads. """
        thread = threading.Thread(target=self.plot_trajectory)
        thread.daemon = True
        thread.start()

    def plot_trajectory(self):
        fig = plt.figure(figsize=(8, 4.5))

        ax3d = fig.add_subplot(2, 3, (1, 3), projection='3d')
        ax3d.view_init(elev=20, azim=150)
        ax_x = fig.add_subplot(2, 3, 4)
        ax_y = fig.add_subplot(2, 3, 5)
        ax_z = fig.add_subplot(2, 3, 6)

        def update(_):
            ax3d.clear()
            ax3d.set_title("Real-Time 3D Trajectory")
            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Z")
            ax3d.set_xlim([-2, 2])
            ax3d.set_ylim([-2, 2])
            ax3d.set_zlim([0, 3])

            x_parrot_3d_plot = self.x_parrot[-400:]
            y_parrot_3d_plot = self.y_parrot[-400:]
            z_parrot_3d_plot = self.z_parrot[-400:]
            x_anafi_3d_plot = self.x_anafi[-400:]
            y_anafi_3d_plot = self.y_anafi[-400:]
            z_anafi_3d_plot = self.z_anafi[-400:]

            if self.x_parrot:
                ax3d.plot(x_parrot_3d_plot, y_parrot_3d_plot, z_parrot_3d_plot, 'ro-', label="Parrot", markersize = 1)

            if self.x_anafi:
                ax3d.plot(x_anafi_3d_plot, y_anafi_3d_plot, z_anafi_3d_plot, 'bo-', label="Anafi", markersize = 1)

            ax3d.legend()

            # ---- 2D XYZ vs Time ----
            min_time = max(0, self.time_data[-1] - 10) if self.time_data else 0

            ax_x.clear()
            ax_y.clear()
            ax_z.clear()

            ax_x.set_title("X vs Time")
            ax_y.set_title("Y vs Time")
            ax_z.set_title("Z vs Time")

            ax_x.set_ylabel("X Position")
            ax_y.set_ylabel("Y Position")
            ax_z.set_ylabel("Z Position")

            ax_x.set_xlabel("Time (s)")
            ax_y.set_xlabel("Time (s)")
            ax_z.set_xlabel("Time (s)")

            ax_x.set_xlim([min_time, min_time + 10])
            ax_y.set_xlim([min_time, min_time + 10])
            ax_z.set_xlim([min_time, min_time + 10])

            ax_x.set_ylim([-3, 3])
            ax_y.set_ylim([-3, 3])
            ax_z.set_ylim([0, 4])

            if self.time_data:
                ax_x.plot(self.time_data, self.x_parrot, 'r-', label="Parrot X")
                ax_y.plot(self.time_data, self.y_parrot, 'g-', label="Parrot Y")
                ax_z.plot(self.time_data, self.z_parrot, 'b-', label="Parrot Z")

                ax_x.plot(self.time_data, self.x_anafi, 'r--', label="Anafi X")
                ax_y.plot(self.time_data, self.y_anafi, 'g--', label="Anafi Y")
                ax_z.plot(self.time_data, self.z_anafi, 'b--', label="Anafi Z")

                ax_x.legend()
                ax_y.legend()
                ax_z.legend()
        
            # plt.pause(0.001)
    
        ani = animation.FuncAnimation(fig, update, interval=50)
        plt.show()


    def update_plot(self):
        """ Updates both 3D trajectory and 2D XYZ vs Time plots. """
        # ---- 3D Trajectory ----
        self.ax3d.clear()
        self.ax3d.set_title("Real-Time 3D Trajectory")
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.set_xlim([-2, 2])
        self.ax3d.set_ylim([-2, 2])
        self.ax3d.set_zlim([0, 3])

        if self.x_parrot:
            self.ax3d.plot(self.x_parrot, self.y_parrot, self.z_parrot, 'ro-', label="Parrot")

        if self.x_anafi:
            self.ax3d.plot(self.x_anafi, self.y_anafi, self.z_anafi, 'bo-', label="Anafi")

        self.ax3d.legend()

        # ---- 2D XYZ vs Time ----
        min_time = max(0, self.time_data[-1] - 10) if self.time_data else 0

        self.ax_x.clear()
        self.ax_y.clear()
        self.ax_z.clear()

        self.ax_x.set_title("X vs Time")
        self.ax_y.set_title("Y vs Time")
        self.ax_z.set_title("Z vs Time")

        self.ax_x.set_ylabel("X Position")
        self.ax_y.set_ylabel("Y Position")
        self.ax_z.set_ylabel("Z Position")
        self.ax_z.set_xlabel("Time (s)")

        self.ax_x.set_xlim([min_time, min_time + 10])
        self.ax_y.set_xlim([min_time, min_time + 10])
        self.ax_z.set_xlim([min_time, min_time + 10])

        self.ax_x.set_ylim([-2, 2])
        self.ax_y.set_ylim([-2, 2])
        self.ax_z.set_ylim([0, 3])

        if self.time_data:
            self.ax_x.plot(self.time_data, self.x_parrot, 'r-', label="Parrot X")
            self.ax_y.plot(self.time_data, self.y_parrot, 'g-', label="Parrot Y")
            self.ax_z.plot(self.time_data, self.z_parrot, 'b-', label="Parrot Z")

            self.ax_x.plot(self.time_data, self.x_anafi, 'r--', label="Anafi X")
            self.ax_y.plot(self.time_data, self.y_anafi, 'g--', label="Anafi Y")
            self.ax_z.plot(self.time_data, self.z_anafi, 'b--', label="Anafi Z")

        self.ax_x.legend()
        self.ax_y.legend()
        self.ax_z.legend()

def main():
    rclpy.init()
    node = RealTimePlot()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()  # Keep ROS2 running
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
