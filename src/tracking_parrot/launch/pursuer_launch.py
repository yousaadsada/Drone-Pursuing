from launch import LaunchDescription
from launch_ros.actions import Node


my_package = "tracking_parrot"
def generate_launch_description():
    tracking_drone_node = Node(
        package= my_package,
        executable='tracking_drone',
        name='tracking_drone_node',
    )

    get_anafi_state_node = Node(
        package= my_package,
        executable='get_anafi_state',
        name='get_anafi_state_node',
    )
    
    get_2d_bbox_node = Node(
        package= my_package,
        executable='get_2d_bbox',
        name='get_2d_bbox_node',
    )

    get_3d_bbox_node = Node(
        package= my_package,
        executable='get_3d_bbox',
        name='get_3d_bbox_node',
    )

    get_3d_pos_node = Node(
        package= my_package,
        executable='get_3d_pos',
        name='get_3d_pos_node',
    )
    

    
    return LaunchDescription([
        tracking_drone_node,
        get_anafi_state_node,
        get_2d_bbox_node,
        get_3d_bbox_node,
        get_3d_pos_node,
   
    ])
