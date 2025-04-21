from setuptools import find_packages, setup

package_name = 'vrx_gym'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/river_follow_env.launch.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jianwen, edison',
    maintainer_email='lijianwen2016@gmail.com, princehan920453024@gmail.com',
    description='Gymnasium Environment of VRX for Vision-driven River Following of ASV',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
