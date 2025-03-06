from setuptools import find_packages, setup

package_name = 'drop_ball'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/get_keypoint_launch.py','launch/drop_ball_launch_yolo.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yousa',
    maintainer_email='yousa@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'collect_jackal_fig = drop_ball.collect_jackal_fig:main',
            'get_anafi_vedio = drop_ball.get_anafi_vedio:main',
            'get_keypoints = drop_ball.get_keypoints:main',
            'get_vicon_data = drop_ball.get_vicon_data:main',
            'moving2points_dropball = drop_ball.moving2points_dropball:main',
            'get_3d_bbox_yolo = drop_ball.get_3d_bbox_yolo:main',
            'get_2d_bbox_yolo = drop_ball.get_2d_bbox_yolo:main',
            'get_3d_pos_yolo = drop_ball.get_3d_pos_yolo:main',
            'drop_ball_jackal_yolo = drop_ball.drop_ball_jackal_yolo:main',
            'get_anafi_state = drop_ball.get_anafi_state:main',
            'get_jackal_state = drop_ball.get_jackal_state:main',
            'camera_test = drop_ball.camera_test:main',
        ],
    },
)
