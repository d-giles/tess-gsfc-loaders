#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Daniel Giles",
    author_email='daniel.k.giles@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Loaders for working with the light curves produced from TESS FFIs by the group at Goddard",
    entry_points={
        'console_scripts': [
            'tess_gsfc_loaders=tess_gsfc_loaders.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='tess_gsfc_loaders',
    name='tess_gsfc_loaders',
    packages=find_packages(include=['tess_gsfc_loaders', 'tess_gsfc_loaders.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/d-giles/tess_gsfc_loaders',
    version='0.1.0',
    zip_safe=False,
)
