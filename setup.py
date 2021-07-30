#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(name='pybsp',
      maintainer='Lyudmil Vladimirov',
      maintainer_email='sglvladi@gmail.com',
      url='https://github.com/sglvladi/PyBSP',
      description='A Python implementation of a Binary Space Partitioning (BSP) tree',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3 :: Only',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
      ],
      packages=find_packages(exclude=('docs', '*.tests')),
      python_requires='>=3.6',
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      use_scm_version=True,
      install_requires=[
          'shapely>=1.7.1', 'networkx>=2.6.1', 'tqdm', 'stonesoup', 'numpy', 'pyshp', 'matplotlib',
          'setuptools>=42',
      ],
      extras_require={},
      )