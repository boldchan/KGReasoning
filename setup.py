# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

# with open('LICENSE') as f:
#     license = f.read()

setup(
    name='tKGR',
    version='0.1.0',
    description='temporal knowledge graph reasoning, package name tKGR is temporary',
    long_description=readme,
#     author='',
#     author_email='me@kennethreitz.com',
#     url='https://github.com/kennethreitz/samplemod',
#     license=license,
    python_requires='~=3.7',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'torch',
        'numpy',
        'importlib'
    ],
    package_data={
        "tKGR": ["data/ICEWS18_forecasting/*.txt"]
    }
)