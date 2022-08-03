#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Analysis simulations: Softmax -- Version 3.3
Last edit:  2022/08/03
Author(s):  Geysen, Steven (SG)
Notes:      - Analysis of simulated data of the task used by
                Marzecova et al. (2019)
            - Release notes:
                * PE boxplot
                
To do:      - Nelder-Mead
            - Explore models
Questions:  
            
Comments:   
            
Sources:    https://elifesciences.org/articles/49547
            https://stackoverflow.com/a/41399555
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
SIM_DIR = OUT_DIR / 'simulations/softmax'



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
# Beta options
beta_options = np.linspace(0.1, 20, 40)
##SG: To have the smallest alpha and beta values in the same corner (left-down)
plotbetas = np.flip(beta_options)

# Switch points
exStruc = pd.read_csv(SIM_DIR / simList[0], index_col='Unnamed: 0')



#%% ~~ Exploration ~~ %%#
#########################


#%% ~~ Optimal thetas ~~ %%#
#--------------------------#


OptimalThetas = np.full((2, 2), np.nan)
negCors = np.zeros((len(alpha_options), len(beta_options), len(MDLS)))

for iti in range(N_ITERS):
    for loca, alphai in enumerate(alpha_options):
        for locb, betai in enumerate(beta_options):
            for locm, modeli in enumerate(MDLS):
                negCors[loca, locb, locm] += sf.sim_negSpearCor((alphai, betai),
                                                                exStruc,
                                                                model=modeli)
negCors /= N_ITERS
# Optimal values
for locm, modeli in enumerate(MDLS):
    toploc = [i[0] for i in np.where(
        negCors[:, :, locm] == np.min(negCors[:, :, locm]))]
    OptimalThetas[locm, :] = [alpha_options[toploc[0]],
                              beta_options[toploc[1]]]
    
    plt.figure(plotnr)
    fig, ax = plt.subplots()
    im, _ = pf.heatmap(np.rot90(negCors[:, :, locm]), np.round(plotbetas, 3),
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

# PE validity effect
pf.pe_validity(MDLS, simList, SIM_DIR, plotnr)
plotnr += 1



#%% ~~ Fitting ~~ %%#
#####################


#%% ~~ Grid search ~~ %%#
#-----------------------#


originalThetas = np.full((len(simList), 2), np.nan)
gridThetas = np.full((len(simList), 2), np.nan)

start_total = time.time()
for simi, filei in enumerate(simList):
    simData = pd.read_csv(SIM_DIR / filei, index_col='Unnamed: 0')
    one_sim = np.zeros((len(alpha_options), len(beta_options)))
    
    # Thetas from simulation
    stringTheta = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", filei)
    otheta = [float(thetai) for thetai in stringTheta]
    originalThetas[simi, :] = otheta[1:]
    
    start_sim = time.time()
    for loca, alphai in enumerate(alpha_options):
        for locb, betai in enumerate(beta_options):
            for iti in range(N_ITERS):
                # one_sim[loca, locb] = sf.sim_negLL((alphai, betai), simData,
                #                                           model='RW')
                one_sim[loca, locb] += sf.sim_negSpearCor((alphai, betai),
                                                                simData,
                                                                model='RW')
            one_sim[loca, locb] /= N_ITERS
            # one_sim[loca, locb] = one_sim[loca, locb] / N_ITERS
    # Optimal values
    toploc = [i[0] for i in np.where(one_sim == np.min(one_sim))]
    gridThetas[simi, :] = [alpha_options[toploc[0]], beta_options[toploc[1]]]
    print(f'Duration sim {simi}: {round((time.time() - start_sim) / 60, 2)} minutes')
    
    if simi % 2 == 0:
        plt.figure(plotnr)
        fig, ax = plt.subplots()
        im, _ = pf.heatmap(np.rot90(one_sim), np.round(plotbetas, 3),
                    np.round(alpha_options, 3), ax=ax,
                    row_name='$\u03B2$', col_name='$\u03B1$',
                    cbarlabel='Negative Spearman Correlation')
        
        plt.suptitle(f'Grid search Negative Spearman Correlation of choice for simulation {simi}')
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
    if simi % 2 == 0:
        print(originalThetas[simi])
        print(initial_guess)
        print(nmThetas[simi])


# ------------------------------------------------------------------------ End
