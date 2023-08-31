#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Recurrent neural network: Simulations -- Version 1
Last edit:  2023/08/31
Author:     Geysen, Steven (SG)
Notes:      - Recurrent neural network for task used by Marzecova et al. (2019)
            - Release notes:
                * Initial commit
To do:      - Build model
            - Train model
            - Test model
Comments:   SG: Trying to understand RNNs on a task that I already understand.
Sources:    
"""



#%% ~~ Imports and directories ~~ %%#


import itertools

import numpy as np
import pandas as pd

import fns.plot_functions as pf
import fns.sim_functions as sf

from natsort import natsorted
from pathlib import Path
from torch import nn


# Directories
SPINE = Path.cwd().parent
OUT_DIR = SPINE / 'results'
if not Path.exists(OUT_DIR):
    Path.mkdir(OUT_DIR)
SIM_DIR = OUT_DIR / 'simulations'
if not Path.exists(SIM_DIR):
    Path.mkdir(SIM_DIR)
SOFT_DIR = SIM_DIR / 'softmax'
if not Path.exists(SOFT_DIR):
    Path.mkdir(SOFT_DIR)
ARG_DIR = SIM_DIR / 'argmax'
if not Path.exists(ARG_DIR):
    Path.mkdir(ARG_DIR)
BIAS_DIR = SIM_DIR / 'biased'
if not Path.exists(BIAS_DIR):
    Path.mkdir(BIAS_DIR)



#%% ~~ Variables ~~ %%#



# ------------------------------------------------------------------------ End
