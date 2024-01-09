from setuptools import setup, find_namespace_packages

setup(
    name='Parametric_Network_Synthesis',
    version='0.0.1',
    packages = find_namespace_packages(where = 'src'),
    package_dir={"": "src"},
    install_requires=['schemdraw', 'numpy', 'matplotlib', 'scikit-rf', 'plotly', 'sympy']
)
