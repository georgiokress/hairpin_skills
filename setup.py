from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hairpin_skills'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Config-Dateien installieren
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        # Launch-Dateien installieren
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='georgio',
    maintainer_email='uvxga@student.kit.edu',
    description='Hairpin Setzprozess Skills fuer SPI-DP',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'test_move = hairpin_skills.test_move:main',
            'test_non_force_skill = hairpin_skills.test_non_force_skill:main',
            'test_touchdown = hairpin_skills.test_touchdown:main',
            'test_sliding_training_day2 = hairpin_skills.test_sliding_training_day2:main',
            'test_sliding = hairpin_skills.test_sliding:main',
            'force_mode_isolated = hairpin_skills.force_mode_isolated:main',
            'force_mode_isolated_auto = hairpin_skills.force_mode_isolated_auto:main',
            'sliding_frame_force_test = hairpin_skills.sliding_frame_force_test:main',
            'radial_sliding_test = hairpin_skills.radial_sliding_test:main',
            'radial_sliding_chamfer_observer = hairpin_skills.radial_sliding_chamfer_observer:main',

        ],
    },
)
# from setuptools import find_packages, setup

# package_name = 'hairpin_skills'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=find_packages(exclude=['test']),
#     data_files=[
#         ('share/ament_index/resource_index/packages',
#             ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='georgio',
#     maintainer_email='uvxga@student.kit.edu',
#     description='TODO: Package description',
#     license='TODO: License declaration',
#     extras_require={
#         'test': [
#             'pytest',
#         ],
#     },
#     entry_points={
#         'console_scripts': [
#         ],
#     },
# )
