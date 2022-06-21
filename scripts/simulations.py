#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Simulations -- Version 1
Last edit:  2022/06/21
Author(s):  Geysen, Steven (SG)
Notes:      - Simulations of the task used by Marzecova et al. (2019)
            - Release notes:
                * Simulations of task
                
To do:      - Grid search
            - Nelder-Mead
            - Explore models
Questions:  
            
Comments:   
            
Sources:    https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00612/full
            https://docs.sympy.org/latest/modules/stats.html
            https://github.com/Kingsford-Group/ribodisomepipeline/blob/master/scripts/exGaussian.py
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html
            https://hannekedenouden.ruhosting.nl/RLtutorial/html/SimulationTopPage.html
            https://lindeloev.shinyapps.io/shiny-rt/
            https://elifesciences.org/articles/49547
"""



#%% ~~ Imports and directories ~~ %%#


import re
import time

import numpy as np
import pandas as pd
import sim_functions as sf

import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats


# Directories
SPINE = Path.cwd().parent
OUT_DIR = SPINE / 'results'
if not Path.exists(OUT_DIR):
    Path.mkdir(OUT_DIR)
SIM_DIR = OUT_DIR / 'simulations'
if not Path.exists(SIM_DIR):
    Path.mkdir(SIM_DIR)



#%% ~~ Variables ~~ %%#


# Number of simulations
N_SIMS = 10
# Plot number
plotnr = 0

# Alpha/eta options
alpha_options = np.linspace(0.01, 1, 40)
# Beta options
beta_options = np.linspace(0.1, 20, 40)



#%% ~~ Data simulation ~~ %%#
#############################


for simi in range(N_SIMS):
    # Create experimental structure
    exStruc = sf.sim_experiment(ppnr=simi)
    
    # Sample data
    ## Select random alpha and beta
    alpha = np.random.choice(alpha_options)
    beta = np.random.choice(beta_options)
    
    Daphne = sf.simRW_1c((alpha, beta), exStruc)
    Hugo = sf.simHybrid_1c((alpha, beta), Daphne)
    
    Hugo.to_csv(SIM_DIR / f'simModels_alpha_{alpha}_beta_{beta}.csv')



#%% ~~ Fitting ~~ %%#
#####################


simList = [filei.name for filei in Path.iterdir(SIM_DIR)]



# ------------------------------------------------------------------------ End
