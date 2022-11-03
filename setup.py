# -*- coding: utf-8 -*-
import os
import sys
from setuptools import setup, find_packages


HERE = os.path.dirname(os.path.abspath((__file__)))

# meta infos
NAME = "dele"
DESCRIPTION = "Implementation of Dynamic Ensemble Architecture Performance Predictor"

with open(os.path.join(os.path.dirname(__file__), "dele", "VERSION")) as f:
    VERSION = f.read().strip()

AUTHOR = "Junbo Zhao"
EMAIL = "zhaojunbo2012@sina.cn"

# package contents
MODULES = []
PACKAGES = find_packages()

# dependencies
INSTALL_REQUIRES = [
    "torch>=1.2.0",
    "numpy",
    "six",
    "scipy"
]


def read_long_description(filename):
    path = os.path.join(HERE, filename)
    if os.path.exists(path):
        return open(path).read()
    return ""


setup(
    name = NAME,
    version = VERSION,
    license = "MIT",
    url = "https://github.com/A-LinCui/DELE",
    author = AUTHOR,
    author_email = EMAIL,
    description = DESCRIPTION,
    long_description = read_long_description("README.md"),
    py_modules = MODULES,
    packages = PACKAGES,
    install_requires = INSTALL_REQUIRES,
    zip_safe = True,
    package_data = {"dele": ["VERSION"]}
)
