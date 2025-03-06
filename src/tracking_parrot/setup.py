from setuptools import find_packages, setup

package_name = 'tracking_parrot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pursuer_launch.py', 'launch/get_keypoint_launch.py','launch/pursuer_launch_yolo.py','launch/pursuer_launch_yolo_speed_prediction.py','launch/pursuer_launch_vicon.py','launch/yolo_detectron_comparison_launch.py']),
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
            'get_keypoints = tracking_parrot.get_keypoints:main',
            'train_model = tracking_parrot.train_model:main',
            'dataset = tracking_parrot.dataset:main',
            'get_2d_bbox = tracking_parrot.get_2d_bbox:main',
            'get_2d_bbox_fix = tracking_parrot.get_2d_bbox_fix:main',
            'get_3d_bbox = tracking_parrot.get_3d_bbox:main',
            'get_3d_bbox_fix = tracking_parrot.get_3d_bbox_fix:main',
            'get_3d_pos = tracking_parrot.get_3d_pos:main',
            'tracking_drone = tracking_parrot.tracking_drone:main',
            'moving2points = tracking_parrot.moving2points:main',
            'get_anafi_state = tracking_parrot.get_anafi_state:main',
            'get_parrot_state = tracking_parrot.get_parrot_state:main',
            'PPO_anafi = tracking_parrot.PPO_anafi:main',
            'collect_parrot_fig = tracking_parrot.collect_parrot_fig:main',
            'get_vicon_data = tracking_parrot.get_vicon_data:main',
            'get_anafi_vedio = tracking_parrot.get_anafi_vedio:main',
            'get_3d_bbox_yolo = tracking_parrot.get_3d_bbox_yolo:main',
            'get_3d_pos_yolo = tracking_parrot.get_3d_pos_yolo:main',
            'tracking_drone_yolo = tracking_parrot.tracking_drone_yolo:main',
            'tracking_drone_vicon = tracking_parrot.tracking_drone_vicon:main',
            'plot_data_yolo = tracking_parrot.plot_data_yolo:main',
            'plot_data_yolo_3d = tracking_parrot.plot_data_yolo_3d:main',
            'test_camera_angle = tracking_parrot.test_camera_angle:main',
            'test_mpc_delay = tracking_parrot.test_mpc_delay:main',
            'tracking_drone_yolo_speed_prediction = tracking_parrot.tracking_drone_yolo_speed_prediction:main',
            'yolo_detectron_comparison = tracking_parrot.yolo_detectron_comparison:main'
        ],
    },
)
