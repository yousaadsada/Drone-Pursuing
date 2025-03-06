from launch import LaunchDescription
from launch_ros.actions import Node


my_package = "drop_ball"
def generate_launch_description():
    drop_ball_node = Node(
        package= my_package,
        executable='drop_ball_jackal_yolo',
        name='drop_ball_node',
    )

    get_anafi_state_node = Node(
        package= my_package,
        executable='get_anafi_state',
        name='get_anafi_state_node',
    )

    get_jackal_state_node = Node(
        package= my_package,
        executable='get_jackal_state',
        name='get_jackal_state_node',
    )

    get_2d_bbox_node = Node(
        package= my_package,
        executable='get_2d_bbox_yolo',
        name='get_2d_bbox_node',
    )
    

    get_3d_bbox_node = Node(
        package= my_package,
        executable='get_3d_bbox_yolo',
        name='get_3d_bbox_node',
    )

    get_3d_pos_node = Node(
        package= my_package,
        executable='get_3d_pos_yolo',
        name='get_3d_pos_node',
    )
    

    
    return LaunchDescription([
        drop_ball_node,
        get_anafi_state_node,
        get_jackal_state_node,
        get_2d_bbox_node,
        get_3d_bbox_node,
        get_3d_pos_node,
   
    ])
