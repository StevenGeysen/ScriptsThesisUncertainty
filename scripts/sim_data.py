#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Simulate data -- Version 1.1
Last edit:  2022/07/17
Author(s):  Geysen, Steven (SG)
Notes:      - Simulations of the task used by Marzecova et al. (2019)
            - Release notes:
                * Fixed indexing parameters
                
To do:      - 
Questions:  
            
Comments:   SG: Only simulations so all analysis happens with the same data.
                This makes comparison of changes in other scripts easier.
            
Sources:    https://elifesciences.org/articles/49547
"""



#%% ~~ Imports and directories ~~ %%#


import numpy as np

import fns.plot_functions as pf
import fns.sim_functions as sf

from pathlib import Path


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



#%% ~~ Variables ~~ %%#


# Number of simulations
N_SIMS = 50
# Plot number
plotnr = 0

# Alpha/eta options
alpha_options = np.linspace(0.01, 1, 40)
# Beta options
beta_options = np.linspace(0.1, 20, 40)

# Create experimental structure to train models on
exStruc = sf.sim_experiment()



#%% ~~ Data simulation ~~ %%#
#############################


for simi in range(N_SIMS):
    # Parameter values
    ## Select random alpha/eta and beta
    alpha = np.random.choice(alpha_options)
    beta = np.random.choice(beta_options)
    
    # Models with argmax policy
    Daphne_arg = sf.simRW_1c((alpha, ), exStruc, asm='arg')
    Hugo_arg = sf.simHybrid_1c((alpha, ), Daphne_arg, asm='arg')
    Wilhelm_arg = sf.simWSLS(Hugo_arg)
    Renee_arg = sf.simRandom(Wilhelm_arg)
    Renee_arg.to_csv(ARG_DIR / f'simData_alpha_{alpha}_argmax.csv')
    
    # pf.selplot(Renee_arg, 'rw', plotnr, thetas=alpha, pp=simi)
    # plotnr += 1
    
    # Models with SoftMax policy
    Daphne_soft = sf.simRW_1c((alpha, beta), exStruc)
    Hugo_soft = sf.simHybrid_1c((alpha, beta), Daphne_soft)
    Wilhelm_soft = sf.simWSLS(Hugo_soft)
    Renee_soft = sf.simRandom(Wilhelm_soft)
    Renee_soft.to_csv(SOFT_DIR / f'simData_alpha_{alpha}_beta_{beta}.csv')
    
    # pf.selplot(Renee_soft, 'rw', plotnr, thetas=(alpha, beta), pp=simi)
    # plotnr += 1



# ------------------------------------------------------------------------ End
