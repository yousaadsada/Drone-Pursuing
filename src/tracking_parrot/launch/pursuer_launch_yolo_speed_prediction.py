from launch import LaunchDescription
from launch_ros.actions import Node
import os

my_package = "tracking_parrot"

def generate_launch_description():
    tracking_drone_node = Node(
        package=my_package,
        executable='tracking_drone_yolo_speed_prediction',
        name='tracking_drone_speed_prediction_node',
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

    get_parrot_state_node = Node(
        package=my_package,
        executable='get_parrot_state',
        name='get_parrot_state_node',
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



    return LaunchDescription([
        tracking_drone_node,
        plot_data_node,
        get_anafi_state_node,
        get_parrot_state_node,
        get_3d_bbox_node,
        get_3d_pos_node,
        # rosbag_record_node,  # ✅ Add rosbag recorder node
    ])
