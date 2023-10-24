from setuptools import setup, find_packages

setup(
    name='tabular_data_augmentation',
    version='1.0.0',
    description='data augmentation for imbalanced datasets',
    author='Tiago F. R. Ribeiro',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'imblearn',
        'sklearn'
    ]
)
