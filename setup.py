# setup.py
from setuptools import setup, find_packages

setup(
    name='cybermicrogrid',
    version='0.1.0',
    description='A library for simulating cyber-physical microgrids and training reinforcement learning agents for grid defense.',
    author='Your Name',
    author_email='your_email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'networkx',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'pandapower',
        'torch-geometric',  # Ensure compatibility and correct package name
        'pydnp3',          # New dependency for DNP3 communications
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
