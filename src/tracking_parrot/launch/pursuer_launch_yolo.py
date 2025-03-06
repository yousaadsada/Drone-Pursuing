from launch import LaunchDescription
from launch_ros.actions import Node
import os

my_package = "tracking_parrot"

def generate_launch_description():
    tracking_drone_node = Node(
        package=my_package,
        executable='tracking_drone_yolo',
        name='tracking_drone_node',
    )

    plot_data_node = Node(
        package=my_package,
        executable='plot_data_yolo',
        name='plot_data_node',
    )

    get_anafi_state_node = Node(
        package=my_package,
        executable='get_anafi_state',
        name='get_anafi_state_node',
    )

    get_3d_bbox_node = Node(
        package=my_package,
        executable='get_3d_bbox_yolo',
        name='get_3d_bbox_node',
    )

    get_3d_pos_node = Node(
        package=my_package,
        executable='get_3d_pos_yolo',
        name='get_3d_pos_node',
    )

    # ✅ Define rosbag recording path
    rosbag_save_path = "/home/yousa/anafi_simulation/data/tracking_parrot_rosbag"
    
    # ✅ Ensure the directory exists
    os.makedirs(rosbag_save_path, exist_ok=True)

    # ✅ Rosbag2 Recording Node
    rosbag_record_node = Node(
        package="rosbag2",
        executable="record",
        name="rosbag_recorder",
        output="screen",
        arguments=["-a", "-o", rosbag_save_path]  # -a (record all topics), -o (output dir)
    )

    return LaunchDescription([
        tracking_drone_node,
        plot_data_node,
        get_anafi_state_node,
        get_3d_bbox_node,
        get_3d_pos_node,
        # rosbag_record_node,  # ✅ Add rosbag recorder node
    ])
