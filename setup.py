#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages, Extension

with open('README.md') as readme_file:
    readme = readme_file.read()

test_requirements = ['pytest>=3', ]
setup_requirements = ['pytest-runner', ]

setup(
    author="Jan N. Fuhg",
    python_requires='>=3',
    classifiers=[
        'Development Status :: 3 - Beta',
        'Intended Audience :: Experts',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    description="Code for model-data-driven yield function modeling",
    install_requires=['numpy', 'torch', 'scipy', 'sklearn', 'sympy', 'matplotlib==3.1.2', 'scikit-image'],
    license="GNU General Public License v3",
    long_description=readme,
    name='modelDataDriven_YieldFunction',
)
