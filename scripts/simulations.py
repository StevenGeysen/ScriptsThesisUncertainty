#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Simulations -- Version 2.2
Last edit:  2022/07/12
Author(s):  Geysen, Steven (SG)
Notes:      - Simulations of the task used by Marzecova et al. (2019)
            - Release notes:
                * Worked on grid search
                
To do:      - Nelder-Mead
            - Explore models
Questions:  
            
Comments:   
            
Sources:    https://elifesciences.org/articles/49547
"""



#%% ~~ Imports and directories ~~ %%#


import re
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import src.plot_functions as pf
import src.sim_functions as sf

from pathlib import Path
from scipy import optimize, stats


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
    exStruc = sf.sim_experiment(simnr=simi)
    
    # Sample data
    ## Select random alpha and beta
    alpha = np.random.choice(alpha_options)
    beta = np.random.choice(beta_options)
    
    Daphne = sf.simRW_1c((alpha, beta), exStruc)
    Hugo = sf.simHybrid_1c((alpha, beta), Daphne)
    Wilhelm = sf.simWSLS(Hugo)
    Renee = sf.simRandom(Wilhelm)
    
    Renee.to_csv(SIM_DIR / f'simModels_alpha_{alpha}_beta_{beta}.csv')
    
    pf.selplot(Renee, 'rw', plotnr, thetas=(alpha, beta), pp=simi)
    plotnr += 1



#%% ~~ Fitting ~~ %%#
#####################


simList = [filei.name for filei in Path.iterdir(SIM_DIR)]



#%% ~~ Grid search ~~ %%#
#-----------------------#


# Smallest alpha and beta values left-below
plotbetas = np.flip(beta_options)
originalThetas = np.full((len(simList), 2), np.nan)
gridThetas = np.full((len(simList), 2), np.nan)

start_total = time.time()
for simi, filei in enumerate(simList):
    simData = pd.read_csv(SIM_DIR / filei, index_col='Unnamed: 0')
    one_totsim_log = np.zeros((len(alpha_options), len(beta_options)))
    
    # Thetas from simulation
    stringTheta = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", filei)
    otheta = [float(thetai) for thetai in stringTheta]
    originalThetas[simi, :] = otheta
    
    start_sim = time.time()
    for loca, alphai in enumerate(alpha_options):
        for locb, betai in enumerate(beta_options):
            # one_totsim_log[loca, locb] = sf.sim_negLL((alphai, betai), simData,
            #                                           model='RW')
            one_totsim_log[loca, locb] = sf.sim_negSpearCor((alphai, betai),
                                                            simData,
                                                            model='RW')
    
    # Optimal values
    maxloc = [i[0] for i in np.where(one_totsim_log == np.min(one_totsim_log))]
    gridThetas[simi, :] = [alpha_options[maxloc[0]], beta_options[maxloc[1]]]
    
    print(f'Duration sim {simi}: {round((time.time() - start_sim) / 60, 2)} minutes')
    
    if simi % 2 == 0:
        plt.figure(plotnr)
        fig, ax = plt.subplots()
        im, _ = pf.heatmap(np.rot90(one_totsim_log), np.round(plotbetas, 3),
                    np.round(alpha_options, 3), ax=ax,
                    row_name='$\u03B2$', col_name='$\u03B1$',
                    cbarlabel='Negative log-likelihood')
        
        plt.suptitle(f'Grid search log-likelihood of choice for simulation {simi}')
        plt.show()
        plotnr += 1
        
        print(originalThetas[simi])
        print(gridThetas[simi])
print(f'Duration total: {round((time.time() - start_total) / 60, 2)} minutes')


##SG: Paired t-test to see if the difference between originalThetas and
    # gridThetas is too big.
## Alphas
print(stats.ttest_rel(originalThetas[:, 0], gridThetas[:, 0]))
## Betas
print(stats.ttest_rel(originalThetas[:, 1], gridThetas[:, 1]))



#%% ~~ Nelder - Mead ~~ %%#
#-------------------------#


nmThetas = np.full((len(simList), 2), np.nan)

for simi, filei in enumerate(simList):
    simData = pd.read_csv(SIM_DIR / filei, index_col='Unnamed: 0')
    initial_guess = (np.random.choice(alpha_options),
                     np.random.choice(beta_options))
    nmThetas[simi, :] = optimize.fmin(sf.sim_negSpearCor, initial_guess,
                                      args = (simData, 'RW'),
                                      ftol = 0.001)



# ------------------------------------------------------------------------ End
