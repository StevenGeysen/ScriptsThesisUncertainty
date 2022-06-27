#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Setup file
Last edit:  2022/06/27
Author(s):  Geysen, Steven (SG)
Notes:      - To import functions
            - Release notes:
                * Initial commit
            
Comments:   
            
Sources:     https://goodresearch.dev/setup.html#install-a-project-package
"""


from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
)