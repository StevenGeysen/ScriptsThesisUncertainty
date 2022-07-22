#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Analysis simulations: Alpha recovery -- Version 1.1.1
Last edit:  2022/07/18
Author(s):  Geysen, Steven (SG)
Notes:      - Analysis of the task used by Marzecova et al. (2019), simulated
                with argmax policy
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

# Number of iterations
N_ITERS = 10
# Models with optimiseable parameters
MDLS = ['RW', 'H']
# Plot number
plotnr = 0

# Alpha/eta options
alpha_options = np.linspace(0.01, 1, 40)

# Switch points
exStruc = pd.read_csv(SIM_DIR / simList[0], index_col='Unnamed: 0')
lag_relCueCol = exStruc.relCueCol.eq(exStruc.relCueCol.shift())
switches = np.where(lag_relCueCol == False)[0][1:]



#%% ~~ Exploration ~~ %%#
#########################


#%% ~~ Optimal thetas ~~ %%#
#--------------------------#


OptimalThetas = np.full((2, 1), np.nan)
negCors = np.zeros((len(alpha_options), len(MDLS)))

for iti in range(N_ITERS):
    for loca, alphai in enumerate(alpha_options):
        for locm, modeli in enumerate(MDLS):
            negCors[loca, locm] += sf.sim_negSpearCor((alphai), exStruc,
                                                      model=modeli)
negCors /= N_ITERS
# Optimal values
for locm, modeli in enumerate(MDLS):
    maxloc = [i[0] for i in np.where(
        negCors[:, locm] == np.min(negCors[:, locm]))]
    OptimalThetas[locm] = [alpha_options[maxloc[0]]]
    
    plt.figure(plotnr)
    fig, ax = plt.subplots()
    im, _ = pf.heatmap(np.rot90(negCors[:, :, locm]), [1],
                np.round(alpha_options, 3), ax=ax,
                row_name='$\u03B2$', col_name='$\u03B1$',
                cbarlabel='Negative Spearman Correlation')
    
    plt.suptitle(f'Negative Spearman Correlation: Optimal values {modeli}')
    plt.show()
    plotnr += 1



#%% ~~ Plots ~~ %%#
#-----------------#


# Learning curve
pf.learning_curve(simList, SIM_DIR, plotnr)
plotnr += 1

# Stay behaviour
pf.p_stay(simList, SIM_DIR, plotnr)
plotnr += 1



#%% ~~ Fitting ~~ %%#
#####################


#%% ~~ Grid search ~~ %%#
#-----------------------#


originalThetas = np.full((len(simList), 1), np.nan)
gridThetas = np.full((len(simList), 1), np.nan)

start_total = time.time()
for simi, filei in enumerate(simList):
    simData = pd.read_csv(SIM_DIR / filei, index_col='Unnamed: 0')
    one_sim = np.zeros((len(alpha_options), 1))
    
    # Thetas from simulation
    stringTheta = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", filei)
    otheta = [float(thetai) for thetai in stringTheta]
    originalThetas[simi] = otheta
    
    start_sim = time.time()
    for loca, alphai in enumerate(alpha_options):
        # one_sim[loca] = sf.sim_negLL((alphai), simData, model='RW')
        one_sim[loca] = sf.sim_negSpearCor((alphai, ), simData, model='RW')
    # Optimal values
    maxloc = [i[0] for i in np.where(one_sim == np.min(one_sim))]
    gridThetas[simi] = alpha_options[maxloc[0]]
    print(f'Duration sim {simi}: {round((time.time() - start_sim) / 60, 2)} minutes')
    
    if simi % 2 == 0:
        plt.figure(plotnr)
        fig, ax = plt.subplots()
        im, _ = pf.heatmap(np.rot90(one_sim), [1],
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
