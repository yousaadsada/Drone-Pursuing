from launch import LaunchDescription
from launch_ros.actions import Node
import os

my_package = "tracking_parrot"

def generate_launch_description():
    yolo_detectron_comparison_node = Node(
        package=my_package,
        executable='yolo_detectron_comparison',
        name='yolo_detectron_comparison_node',
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

    get_3d_bbox_node_yolo = Node(
        package=my_package,
        executable='get_3d_bbox_yolo',
        name='get_3d_bbox_node',
    )

    get_3d_pos_node_yolo = Node(
        package=my_package,
        executable='get_3d_pos_yolo',
        name='get_3d_pos_node',
    )

    get_2d_bbox_node_detectron = Node(
        package= my_package,
        executable='get_2d_bbox',
        name='get_2d_bbox_node',
    )

    get_3d_bbox_node_detectron = Node(
        package= my_package,
        executable='get_3d_bbox',
        name='get_3d_bbox_node',
    )

    get_3d_pos_node_detectron = Node(
        package= my_package,
        executable='get_3d_pos',
        name='get_3d_pos_node',
    )
    


    return LaunchDescription([
        yolo_detectron_comparison_node,
        plot_data_node,
        get_anafi_state_node,
        get_3d_bbox_node_yolo,
        get_3d_pos_node_yolo,
        get_2d_bbox_node_detectron,
        get_3d_bbox_node_detectron,
        get_3d_pos_node_detectron
        # rosbag_record_node,  # ✅ Add rosbag recorder node
    ])
