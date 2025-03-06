import rclpy
from rclpy.node import Node
import matplotlib
matplotlib.use("TkAgg")  # Ensure Matplotlib runs properly
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import threading
from anafi_msg.msg import PlotData  # Ensure correct ROS message
from rclpy.executors import MultiThreadedExecutor

class RealTimePlot_3d(Node):
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

        # Start Matplotlib plots in separate threads
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
        thread1 = threading.Thread(target=self.plot_3d_trajectory)
        thread2 = threading.Thread(target=self.plot_xyz_vs_time)
        
        thread1.daemon = True
        thread2.daemon = True
        
        thread1.start()
        #thread2.start()

    def plot_3d_trajectory(self):
        """ 3D Plot for real-time XYZ trajectory. """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def update(_):
            ax.clear()
            ax.set_title("Real-Time 3D Trajectory")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            ax.set_xlim([-2, 2])  # X range
            ax.set_ylim([-2, 2])  # Y range
            ax.set_zlim([0, 3])   # Z range


            # Plot Parrot Drone Trajectory
            if self.x_parrot:
                ax.plot(self.x_parrot, self.y_parrot, self.z_parrot, 'ro-', label="Parrot")
            
            # Plot Anafi Drone Trajectory
            if self.x_anafi:
                ax.plot(self.x_anafi, self.y_anafi, self.z_anafi, 'bo-', label="Anafi")

            ax.legend()
            plt.pause(0.001)

        ani = animation.FuncAnimation(fig, update, interval=50)
        plt.show()

    def plot_xyz_vs_time(self):
        """ 2D Plots for X, Y, Z vs Time. """
        fig, axs = plt.subplots(3, 1, figsize=(8, 6))

        def update(_):
            axs[0].clear()
            axs[1].clear()
            axs[2].clear()

            axs[0].set_title("X vs Time")
            axs[1].set_title("Y vs Time")
            axs[2].set_title("Z vs Time")

            axs[0].set_ylabel("X Position")
            axs[1].set_ylabel("Y Position")
            axs[2].set_ylabel("Z Position")
            axs[2].set_xlabel("Time (s)")


            axs[0].set_xlim([0, 10])  # X range
            axs[1].set_xlim([0, 10])  # Y range
            axs[2].set_xlim([0, 10])   # Z range

            axs[0].set_ylim([-2, 2])  # X range
            axs[1].set_ylim([-2, 2])  # Y range
            axs[2].set_ylim([0, 3])   # Z range

            # Plot Parrot drone
            if self.time_data:
                min_time = max(0, self.time_data[-1] - 10)  # Keep the last 10 seconds
                axs[0].set_xlim([min_time, min_time + 10])
                axs[1].set_xlim([min_time, min_time + 10])
                axs[2].set_xlim([min_time, min_time + 10])

                axs[0].set_ylim([-2, 2])  # X range
                axs[1].set_ylim([-2, 2])  # Y range
                axs[2].set_ylim([0, 3])   # Z range

                # Plot Parrot drone
                axs[0].plot(self.time_data, self.x_parrot, 'r-', label="Parrot X")
                axs[1].plot(self.time_data, self.y_parrot, 'g-', label="Parrot Y")
                axs[2].plot(self.time_data, self.z_parrot, 'b-', label="Parrot Z")

                # Plot Anafi drone
                axs[0].plot(self.time_data, self.x_anafi, 'r--', label="Anafi X")
                axs[1].plot(self.time_data, self.y_anafi, 'g--', label="Anafi Y")
                axs[2].plot(self.time_data, self.z_anafi, 'b--', label="Anafi Z")

                axs[0].legend()
                axs[1].legend()
                axs[2].legend()

            plt.pause(0.001)

        ani = animation.FuncAnimation(fig, update, interval=50)
        plt.show()

def main():
    rclpy.init()
    node = RealTimePlot_3d()

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
