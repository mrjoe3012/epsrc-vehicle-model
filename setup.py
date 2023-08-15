from setuptools import find_packages, setup

setup(
    name='epsrc_vehicle_model',
    packages=find_packages(include=['epsrc_vehicle_model']),
    version='1.0.0',
    description='Neural Network Vehicle Model',
    author='Joseph Agrane',
    setup_requires=['torch', 'scipy', 'tqdm', 'scipy'],
    #license='',
)
