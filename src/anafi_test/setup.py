from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'real_world'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='labadmin',
    maintainer_email='labadmin@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'get_jackal_state = real_world.get_jackal_state:main',
            'get_intrinsic_matrix = real_world.get_intrinsic_matrix:main',   
            'get_drone_state = real_world.get_drone_state:main',
            'get_drone_state02 = real_world.get_drone_state02:main',
            'get_drone_state_sim = real_world.get_drone_state_sim:main',
            'pid_moving2point = real_world.pid_moving2point:main',       
            'collect_real_drone_data = real_world.collect_real_drone_data:main',
            'manual_control = real_world.manual_control:main',
            'collect_real_drone_data_vicon = real_world.collect_real_drone_data_vicon:main',
            'collect_real_drone_data_fourier_filter = real_world.collect_real_drone_data_fourier_filter:main',
            'mpc_control_drone_imu = real_world.mpc_control_drone_imu:main',
            'mpc_control_drone_imu_changed = real_world.mpc_control_drone_imu_changed:main',
            'mpc_control_vicon_speed = real_world.mpc_control_vicon_speed:main',
            'mpc_control_pytorch = real_world.mpc_control_pytorch:main',
            'mpc_control_drone_pytorch_plus = real_world.mpc_control_drone_pytorch_plus:main',
            'calculate_state_fcn = real_world.calculate_state_fcn:main',
            'cv_stream = real_world.cv_stream:main',
            'get_3d_pos_apriltag = real_world.get_3d_pos_apriltag:main',
            'get_3D_bbox = real_world.get_3D_bbox:main',
            'get_2D_bbox = real_world.get_2D_bbox:main',
            'get_af_frame = real_world.get_af_frame:main',
            'get_drone_pnp = real_world.get_drone_pnp:main',
            'april_tag = real_world.april_tag:main',
            'track_arcuo = real_world.track_arcuo:main',
            'mpc_anafi = real_world.mpc_anafi:main',
            
        ],
    },
)