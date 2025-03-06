from launch import LaunchDescription
from launch_ros.actions import Node


my_package = "drop_ball"
def generate_launch_description():
    get_vicon_data_node = Node(
        package= my_package,
        executable='get_vicon_data',
        name='get_vicon_data_node',
    )

    get_anafi_vedio_node = Node(
        package= my_package,
        executable='get_anafi_vedio',
        name='get_anafi_vedio_node',
    )
    
    get_keypoints_node = Node(
        package= my_package,
        executable='get_keypoints',
        name='get_keypoints_node',
    )



    
    return LaunchDescription([
        get_vicon_data_node,
        get_anafi_vedio_node,
        get_keypoints_node,
  
   
    ])
