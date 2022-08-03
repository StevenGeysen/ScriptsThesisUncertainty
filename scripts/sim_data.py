#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Simulate data -- Version 1.2
Last edit:  2022/08/03
Author(s):  Geysen, Steven (SG)
Notes:      - Simulations of the task used by Marzecova et al. (2019)
            - Release notes:
                * More simulations with same parameter values
                
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
N_SIMS = 5
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
    for alphai in alpha_options:
        # Parameter values
        ## Select random alpha/eta and beta
        beta = np.random.choice(beta_options)
        title = f'simData_sim{simi}_alpha_{alphai}'
        
        # Models with argmax policy
        Daphne_arg = sf.simRW_1c((alphai, ), exStruc, asm='arg')
        Hugo_arg = sf.simHybrid_1c((alphai, ), Daphne_arg, asm='arg')
        Wilhelm_arg = sf.simWSLS(Hugo_arg)
        Renee_arg = sf.simRandom(Wilhelm_arg)
        Renee_arg.to_csv(ARG_DIR / (title + '_argmax.csv'))
        
        # pf.selplot(Renee_arg, 'rw', plotnr, thetas=alphai, pp=simi)
        # plotnr += 1
        
        # Models with SoftMax policy
        Daphne_soft = sf.simRW_1c((alphai, beta), exStruc)
        Hugo_soft = sf.simHybrid_1c((alphai, beta), Daphne_soft)
        Wilhelm_soft = sf.simWSLS(Hugo_soft)
        Renee_soft = sf.simRandom(Wilhelm_soft)
        Renee_soft.to_csv(SOFT_DIR / (title + f'_beta_{beta}_softmax.csv'))
        
        # pf.selplot(Renee_soft, 'rw', plotnr, thetas=(alphai, beta), pp=simi)
        # plotnr += 1



# ------------------------------------------------------------------------ End
