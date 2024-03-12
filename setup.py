from setuptools import setup, find_namespace_packages

setup(
    name='Parametric_Network_Synthesis',
    version='0.5.0',
    packages = find_namespace_packages(where = 'src'),
    package_dir={"": "src"},
    install_requires=['setuptools',
                      'schemdraw',
                      'numpy',
                      'matplotlib',
                      'scikit-rf',
                      'plotly',
                      'sympy',
                      'ipython',
                      'ipywidgets',
                      'tqdm',
                      'pandas',
                      'scipy',
                      'pyqt5',
                      'proplot==0.9.5']
)