from setuptools import setup, find_packages

setup(
    name='Parametric_Network_Synthesis',
    version='0.0.1',
    packages = find_packages(where = 'src', include = '*tools')
)