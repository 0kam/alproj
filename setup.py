# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py
import os, sys
from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements

setup(
    name='alproj',
    version='0.1.0',
    description='A georectification tool for alpine landscape photographs, written in Python',
    long_description=readme,
    author='OKAMOTO, Ryotaro',
    author_email='okamoto@pe.ska.life.tsukuba.ac.jp',
    url='https://github.com/0kam/alproj
    ',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
    install_requires=read_requirements()
)

