'''
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache 2.0 License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache 2.0 License for more details.
'''

from pkg_resources import parse_requirements
from setuptools import find_packages, setup
 
def load_requirements(file):
    """Parse requirements from file"""
    with open(file, "r") as reqs:
        return [str(req) for req in parse_requirements(reqs)]

setup(
    name='flsuite',
    packages=find_packages(),
    version='0.1.0',
    install_requires=load_requirements('./requirements.txt'),
)