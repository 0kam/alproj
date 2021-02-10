# -*- coding: utf-8 -*-

import os, sys
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="alproj",
    version="0.1.0",
    license=license,
    description="A georectification tool for alpine landscape photographs, written in Python",
    author="Ryotaro Okamoto",
    url="https://github.com/0kam/alproj",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file('requirements.txt'),
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"]
)

