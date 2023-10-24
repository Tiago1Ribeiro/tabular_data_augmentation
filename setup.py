from setuptools import setup, find_packages

setup(
    name='data augmentation',
    version='1.0.0',
    description='Data augmentation for imbalanced datasets',
    author='Tiago F. R. Ribeiro',,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'imblearn',
        'sklearn'
    ]
)