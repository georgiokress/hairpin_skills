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
            'radial_sliding_test = hairpin_skills.radial_sliding_test:main',
            'phase_b_spi_data_collection = hairpin_skills.phase_b_spi_data_collection:main',
            'ati_rdt_wrench = hairpin_skills.ati_rdt_wrench:main',
        ],
    },
)
