#!/usr/bin/env python
"""Distribution configuration."""

from setuptools import setup, find_packages

with open("README.rst", encoding='utf-8') as f:
    README = f.read()

setup(
    name='voxcell',
    author="Blue Brain Project, EPFL",
    description='Voxcell is a small library to handle probability'
                ' distributions that have a spatial component and to use them'
                ' to build collection of cells in space.',
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/BlueBrain/voxcell",
    download_url="https://github.com/BlueBrain/voxcell",
    license='Apache-2',
    install_requires=[
        'h5py>=3.1.0',
        'numpy>=1.9',
        'pandas>=0.24.2',
        'pynrrd>=0.4.0',
        'requests>=2.18',
        'scipy>=1.2.0',
    ],
    packages=find_packages(),
    python_requires='>=3.7',

    setup_requires=[
        'setuptools_scm',
    ],
    use_scm_version={
        "local_scheme": "no-local-version",
    },

    extras_require={
        'docs': [
            'sphinx-bluebrain-theme',
        ],
    },

    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
