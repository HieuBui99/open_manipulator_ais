from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'camera_calibrate'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        # (os.path.join('share', package_name, package_name), glob('apriltag.py')),
        (os.path.join('share', package_name, 'rviz_config'), glob('rviz_config/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
             'extrinsics_estimator = camera_calibrate.extrinsics_estimator:main',
             'visualize_chessboard = camera_calibrate.visualize_chessboard:main',
            'box_tf_publish = camera_calibrate.box_tf_publish:main',
            'object_pose_tf = camera_calibrate.object_pose_tf:main',
        ],
    },
)
