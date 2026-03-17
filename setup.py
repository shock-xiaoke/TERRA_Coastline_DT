"""Setup configuration for TERRA UGLA package."""

import os

from setuptools import find_packages, setup

with open('readme.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

requirements_file = 'requirements.in' if os.path.exists('requirements.in') else 'requirements.txt'
with open(requirements_file, 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='terra-ugla',
    version='1.0.0',
    description='Coastal Vegetation Edge Detection System',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='TERRA UGLA Team',
    python_requires='>=3.8',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
