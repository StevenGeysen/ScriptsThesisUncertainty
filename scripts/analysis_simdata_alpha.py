#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Analysis simulations: Alpha recovery -- Version 2.1
Last edit:  2022/08/05
Author(s):  Geysen, Steven (SG)
Notes:      - Analysis of the task used by Marzecova et al. (2019), simulated
                with argmax policy
            - Release notes:
                * PE validity effect
                * 1 dimensional grid search
                
To do:      - Nelder-Mead
            - Explore models
            - Performance plots (box 2 - figure 1.A)
Questions:  
            
Comments:   
            
Sources:    https://elifesciences.org/articles/49547
"""



#%% ~~ Imports and directories ~~ %%#


import re
import time

import numpy as np
import pandas as pd

import fns.plot_functions as pf
import fns.sim_functions as sf

from pathlib import Path
from scipy import optimize, stats


# Directories
SPINE = Path.cwd().parent
OUT_DIR = SPINE / 'results'
SIM_DIR = OUT_DIR / 'simulations/argmax'



#%% ~~ Variables ~~ %%#


# Filenames of simulated data
simList = [filei.name for filei in Path.iterdir(SIM_DIR)]
simList = simList[:20]  ##SG: First few to test everything quickly.
# Experimental structure
exStruc = pd.read_csv(SIM_DIR / simList[0], index_col='Unnamed: 0')

# Number of iterations
N_ITERS = 10
# Models with optimiseable parameters
MDLS = ['RW', 'H']

# Alpha/eta options
alpha_options = np.linspace(0.01, 1, 40)

# Plot number
plotnr = 0




#%% ~~ Exploration ~~ %%#
#########################


#%% ~~ Optimal thetas ~~ %%#
#--------------------------#


OptimalThetas = np.full((2, 1), np.nan)
negCors = np.zeros((len(alpha_options), len(MDLS)))

for locm, modeli in enumerate(MDLS):
    for loca, alphai in enumerate(alpha_options):
        for iti in range(N_ITERS):
            negCors[loca, locm] += sf.sim_negSpearCor([alphai], exStruc,
                                                      model=modeli)
    # Optimal values
    modeldata = negCors[:, locm] / N_ITERS
    toploc = [i[0] for i in np.where(modeldata == np.min(modeldata))]
    if len(toploc) > 1:
        print(toploc)
    OptimalThetas[locm] = [alpha_options[toploc[0]]]
    
    pf.heatmap_1d(modeli, modeldata, toploc, alpha_options, plotnr)
    plotnr += 1



#%% ~~ Plots ~~ %%#
#-----------------#


# Learning curve
pf.learning_curve(simList, SIM_DIR, plotnr)
plotnr += 1

# Stay behaviour
pf.p_stay(simList, SIM_DIR, plotnr)
plotnr += 1

# PE validity effect
pf.pe_validity(MDLS, simList, SIM_DIR, plotnr)
plotnr += 1



#%% ~~ Fitting ~~ %%#
#####################


#%% ~~ Grid search ~~ %%#
#-----------------------#


originalThetas = np.full((len(simList), 1), np.nan)
gridThetas = np.full((len(simList), len(MDLS)), np.nan)

start_total = time.time()
for simi, filei in enumerate(simList):
    simData = pd.read_csv(SIM_DIR / filei, index_col='Unnamed: 0')
    one_sim = np.zeros((len(alpha_options), len(MDLS)))
    
    # Thetas from simulation
    stringTheta = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", filei)
    otheta = [float(thetai) for thetai in stringTheta][-1]
    originalThetas[simi] = otheta
    
    start_sim = time.time()
    for locm, modeli in enumerate(MDLS):
        for loca, alphai in enumerate(alpha_options):
            for iti in range(N_ITERS):
                # one_sim[loca, locm] += sf.sim_negLL((alphai, ), simData, model='RW')
                one_sim[loca, locm] += sf.sim_negSpearCor((alphai, ), simData,
                                                     model=modeli)
        modeldata = one_sim[:, locm] / N_ITERS
        # Optimal values
        toploc = [i[0] for i in np.where(modeldata == np.min(modeldata))]
        gridThetas[simi, locm] = alpha_options[toploc[0]]
    print(f'Duration sim {simi}: {round((time.time() - start_sim) / 60, 2)} minutes')
    
    if simi % 2 == 0:
        pf.heatmap_1d(modeli, modeldata, toploc, alpha_options, plotnr,
                      gridThetas[simi])
        plotnr += 1
        
        print(originalThetas[simi])
        print(gridThetas[simi])
print(f'Duration total: {round((time.time() - start_total) / 60, 2)} minutes')


##SG: Paired t-test to see if the difference between originalThetas and
    # gridThetas is too big.
## Alphas
print(stats.ttest_rel(originalThetas[:, 0], gridThetas[:, 0]))
## Etas
print(stats.ttest_rel(originalThetas[:, 0], gridThetas[:, 1]))



#%% ~~ Nelder - Mead ~~ %%#
#-------------------------#


nmThetas = np.full((len(simList), 1), np.nan)

for simi, filei in enumerate(simList):
    simData = pd.read_csv(SIM_DIR / filei, index_col='Unnamed: 0')
    initial_guess = np.random.choice(alpha_options)
    nmThetas[simi] = optimize.fmin(sf.sim_negSpearCor, initial_guess,
                                   args = (simData, 'RW'),
                                   ftol = 0.001)
    if simi % 2 == 0:
        print(originalThetas[simi])
        print(initial_guess)
        print(nmThetas[simi])



# ------------------------------------------------------------------------ End
