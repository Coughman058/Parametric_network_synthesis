from setuptools import setup

setup(
    name='Parametric_Network_Synthesis',
    version='0.0.1',
    packages = ['parametricSynthesis',
                'parametricSynthesis.network_tools',
                'parametricSynthesis.drawing_tools'],
    package_dir = {'parametricSynthesis': 'parametricSynthesis'}
)