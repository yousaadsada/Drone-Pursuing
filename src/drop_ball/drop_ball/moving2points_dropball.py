import tkinter as tk
import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading

# Replace this with the IP address of your Arduino web server
ARDUINO_IP = "http://10.42.0.12"  # Replace with your Arduino's IP address
SERVO_ENDPOINT = "/servo?angle="


class GripperControlApp:
    def __init__(self, root, ros_node):
        self.root = root
        self.ros_node = ros_node
        self.root.title("Gripper Control")

        # Open Button
        open_button = tk.Button(self.root, text="Open Gripper", command=self.open_gripper, height=2, width=20)
        open_button.pack(pady=10)

        # Close Button
        close_button = tk.Button(self.root, text="Close Gripper", command=self.close_gripper, height=2, width=20)
        close_button.pack(pady=10)

        # Quit Button
        quit_button = tk.Button(self.root, text="Quit", command=self.quit_program, height=2, width=20)
        quit_button.pack(pady=10)

    def open_gripper(self):
        self.send_command(160)  # Angle for "Open"

    def close_gripper(self):
        self.send_command(180)  # Angle for "Close"

    def send_command(self, angle):
        try:
            url = f"{ARDUINO_IP}{SERVO_ENDPOINT}{angle}"
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Command sent successfully: {angle}")
            else:
                print(f"Failed to send command. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error sending command: {e}")

    def quit_program(self):
        print("Exiting program")
        self.ros_node.destroy_node()
        rclpy.shutdown()
        self.root.destroy()


class GripperControlNode(Node):
    def __init__(self):
        super().__init__('gripper_control_node')
        self.subscription = self.create_subscription(String, 'gripper_command', self.command_callback, 10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info("Gripper Control Node Initialized")

    def command_callback(self, msg):
        command = msg.data.lower()
        if command == "open":
            self.send_command(160)  # Angle for "Open"
        elif command == "close":
            self.send_command(180)  # Angle for "Close"
        else:
            self.get_logger().warning(f"Unknown command received: {command}")

    def send_command(self, angle):
        try:
            url = f"{ARDUINO_IP}{SERVO_ENDPOINT}{angle}"
            response = requests.get(url)
            if response.status_code == 200:
                self.get_logger().info(f"Command sent successfully: {angle}")
            else:
                self.get_logger().error(f"Failed to send command. Status code: {response.status_code}")
        except Exception as e:
            self.get_logger().error(f"Error sending command: {e}")


def ros_spin_thread(node):
    rclpy.spin(node)


def main():
    rclpy.init()

    # Create ROS2 Node
    ros_node = GripperControlNode()

    # Start ROS2 spin in a separate thread
    spin_thread = threading.Thread(target=ros_spin_thread, args=(ros_node,))
    spin_thread.start()

    # Create the GUI
    root = tk.Tk()
    app = GripperControlApp(root, ros_node)
    root.mainloop()

    # Join the ROS2 spin thread on exit
    spin_thread.join()


if __name__ == "__main__":
    main()
