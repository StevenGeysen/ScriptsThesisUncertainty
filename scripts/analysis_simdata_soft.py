#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Analysis simulations: Softmax -- Version 4.2.1
Last edit:  2022/09/13
Author(s):  Geysen, Steven (SG)
Notes:      - Analysis of simulated data of the task used by
                Marzecova et al. (2019)
            - Release notes:
                * Accuracy
                * Optimal values Nelder-Mead
                
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

import fns.assisting_functions as af
import fns.plot_functions as pf
import fns.sim_functions as sf
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import optimize, stats


# Directories
SPINE = Path.cwd().parent
OUT_DIR = SPINE / 'results'
SIM_DIR = OUT_DIR / 'simulations/softmax'
OPT_DIR = SIM_DIR / 'optimal'
if not Path.exists(OPT_DIR):
    Path.mkdir(OPT_DIR)


#%% ~~ Variables ~~ %%#


# Filenames of simulated data
simList = [filei.name for filei in Path.iterdir(SIM_DIR)
           if filei.name.endswith('csv')]
simList = simList[:50]  ##SG: First few to test everything quickly.
# Experimental structure
exStruc = pd.read_csv(SIM_DIR / simList[0], index_col='Unnamed: 0')

ALL_MDLS = ['RW', 'H', 'W', 'R']
# Models with optimiseable parameters
MDLS = ['RW', 'H']
# Number of iterations
N_ITERS = 50
# Number of simulations
N_SIMS = 100
# Number of trials in bin
binsize = 15

# Alpha/eta options
alpha_options = np.linspace(0.01, 1, 20)
# Beta options
beta_options = np.linspace(0.1, 20, 20)

# Plot specs
##SG: To have the smallest alpha and beta values in the same corner (left-down)
plotbetas = np.flip(beta_options)
## Plot number
plotnr = 0
## Model labels
models = af.labelDict()
## Switch points
lag_relCueCol = exStruc.relCueCol.eq(exStruc.relCueCol.shift())
switches = np.where(lag_relCueCol == False)[0]



#%% ~~ Exploration ~~ %%#
#########################


#%% ~~ Optimal thetas ~~ %%#
#--------------------------#


fitStruc = sf.sim_experiment()
OptimalThetas = np.full((2, 2), np.nan)
negCors = np.zeros((len(alpha_options), len(beta_options), len(MDLS)))

for iti in range(N_ITERS):
    for loca, alphai in enumerate(alpha_options):
        for locb, betai in enumerate(beta_options):
            for locm, modeli in enumerate(MDLS):
                negCors[loca, locb, locm] += sf.sim_negSpearCor((alphai, betai),
                                                                fitStruc,
                                                                model=modeli)
negCors /= N_ITERS
# Optimal values
for locm, modeli in enumerate(MDLS):
    toploc = [i[0] for i in np.where(
        negCors[:, :, locm] == np.min(negCors[:, :, locm]))]
    OptimalThetas[locm, :] = [alpha_options[toploc[0]],
                              beta_options[toploc[1]]]
    
    fig, ax = plt.subplots()
    im, _ = pf.heatmap(np.rot90(negCors[:, :, locm]), np.round(plotbetas, 3),
                np.round(alpha_options, 3), ax=ax,
                row_name='$\u03B2$', col_name='$\u03B1$',
                cbarlabel='Negative Spearman Correlation')
    
    plt.suptitle(f'Negative Spearman Correlation: Optimal values {modeli}')
    plt.show()
    plotnr += 1

title = f'sim_gridsearch_{N_ITERS}iters_softmax_optimal.csv'
pd.DataFrame(OptimalThetas, columns=MDLS).to_csv(OUT_DIR / 'simulations' / title)



#%% ~~ Nelder - Mead ~~ %%#


initial_guess = np.full((N_ITERS, 2), np.nan)
nmThetas = np.zeros((len(MDLS), 2))

start_total = time.time()
for iti in range(N_ITERS):
    start_iti = time.time()
    initial_guess[iti, :] = (np.random.choice(alpha_options),
                             np.random.choice(beta_options))
    for locm, modeli in enumerate(MDLS):
        nmThetas[locm, :] += optimize.fmin(
            sf.sim_negSpearCor, initial_guess[iti, :],
            args=(fitStruc, modeli), ftol=0.001)
    print(f'Duration {iti}: {round((time.time() - start_iti) / 60, 2)} minutes')
print(f'Duration total: {round((time.time() - start_total) / 60, 2)} minutes')

nmThetas /= N_ITERS
# Save optimal values
title = f'sim_NelderMead_{N_ITERS}iters_softmax_optimal.csv'
pd.DataFrame(nmThetas, columns=MDLS).to_csv(OUT_DIR / 'simulations' / title)



#%% ~~ Optimal simulations ~~ %%#


for simi in range(N_SIMS):
    Daphne = sf.simRW_1c(OptimalThetas[0, :], fitStruc)
    Hugo = sf.simHybrid_1c(OptimalThetas[1, :], Daphne)
    Wilhelm = sf.simWSLS(Hugo)
    Renee = sf.simRandom(Wilhelm)
    Renee.to_csv(OPT_DIR / f'sim{simi}_softmax_optimal.csv')
    
    # pf.selplot(Renee, 'rw', plotnr, thetas=(alphai, beta), pp=simi)
    # plotnr += 1



#%% ~~ Plots ~~ %%#
#-----------------#


# Learning curve
pf.learning_curve(simList, SIM_DIR, plotnr)
plotnr += 1

# Stay behaviour
pf.p_stay(simList, SIM_DIR, plotnr)
plotnr += 1

# PE validity effect
pf.pe_validity(MDLS, simList, SIM_DIR)
plotnr += 1

# PE over time
## PE curve
pf.pe_curve(MDLS, simList, SIM_DIR, plotnr)
plotnr += 1
for modeli in MDLS:
    pf.pe_curve(modeli, simList, SIM_DIR, plotnr)
    plotnr += 1

## Binned PE
firstDict = {keyi: [] for keyi in MDLS}
lastDict = {keyi: [] for keyi in MDLS}

for simi, filei in enumerate(simList):
    simData = pd.read_csv(SIM_DIR / filei, index_col='Unnamed: 0')
    ## Bin RT of first and last b trials
    for modeli in MDLS:
        sim_pe = simData[f'RPE_{modeli}']
        for starti, endi in af.pairwise(switches):
            firstDict[modeli].append(np.nanmean(sim_pe[starti:
                                                       starti + binsize]))
            lastDict[modeli].append(np.nanmean(sim_pe[endi - binsize:
                                                      endi + 1]))

for modeli in MDLS:
    ## Remove NaN values
    firstlist = [i for i in firstDict[modeli] if np.isnan(i) == False]
    lastlist = [i for i in lastDict[modeli] if np.isnan(i) == False]
    plotbins = [firstlist, lastlist]
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    fig.suptitle(f'Mean binned PE {models[modeli]}, SoftMax')
    vplot, bplot = axs[0], axs[1]
    ## Violin plot
    vplot.violinplot(plotbins,
                      showmeans=False,
                      showmedians=True)
    vplot.set_title('Violin plot simulations')
    ## Box plot
    bplot.boxplot(plotbins)
    bplot.set_title('Box plot simulations')
    
    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xticks([y + 1 for y in range(len(plotbins))],
                      labels=[f'First {binsize} trials',
                              f'Last {binsize} trials'])
        ax.set_xlabel('Trial numbers')
        ax.set_ylabel('Estimated PEs')
    
    plt.show()
    plotnr += 1



#%% ~~ Fitting ~~ %%#
#####################


#%% ~~ Grid search ~~ %%#
#-----------------------#


originalThetas = np.full((len(simList), 2), np.nan)
gridThetas = np.full((len(simList), len(MDLS), 2), np.nan)

start_total = time.time()
for simi, filei in enumerate(simList):
    simData = pd.read_csv(SIM_DIR / filei, index_col='Unnamed: 0')
    one_sim = np.zeros((len(alpha_options), len(beta_options), len(MDLS)))
    
    # Thetas from simulation
    stringTheta = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", filei)
    otheta = [float(thetai) for thetai in stringTheta]
    originalThetas[simi, :] = otheta[1:]
    
    start_sim = time.time()
    for locm, modeli in enumerate(MDLS):
        for loca, alphai in enumerate(alpha_options):
            for locb, betai in enumerate(beta_options):
                for iti in range(N_ITERS):
                    # one_sim[loca, locb, locm] = sf.sim_negLL(
                    #     (alphai, betai), simData, model=modeli, asm='soft'
                    #     )
                    one_sim[loca, locb, locm] += sf.sim_negSpearCor(
                        (alphai, betai), simData, model=modeli, asm='soft'
                        )
        modeldata = one_sim[:, :, locm] / N_ITERS
    # Optimal values
    toploc = [i[0] for i in np.where(modeldata == np.min(one_sim))]
    gridThetas[simi, locm, :] = [alpha_options[toploc[0]], beta_options[toploc[1]]]
    print(f'Duration sim {simi}: {round((time.time() - start_sim) / 60, 2)} minutes')
    
    # Intermittent checking
    if simi % 2 == 0:
        fig, ax = plt.subplots()
        im, _ = pf.heatmap(np.rot90(one_sim), np.round(plotbetas, 3),
                    np.round(alpha_options, 3), ax=ax,
                    row_name='$\u03B2$', col_name='$\u03B1$',
                    cbarlabel='Negative Spearman Correlation')
        
        plt.suptitle(f'Grid search Negative Spearman Correlation of choice {simi}')
        plt.show()
        plotnr += 1
        
        print(originalThetas[simi])
        print(gridThetas[simi])
print(f'Duration total: {round((time.time() - start_total) / 60, 2)} minutes')

# Save optimal values
for locm, modeli in enumerate(MDLS):
    title = f'sim_gridsearch_{N_ITERS}iters_softmax_{modeli}.csv'
    pd.DataFrame(gridThetas[:, locm, :], columns=MDLS).to_csv(OUT_DIR / title)


for locm, modeli in enumerate(MDLS):
    print(modeli)
    ##SG: Paired t-test to see if the difference between originalThetas and
        # gridThetas is too big.
    print('Alpha/eta:', stats.ttest_rel(originalThetas[:, 0],
                                        gridThetas[:, locm, 0],
                                        nan_policy='omit'))
    print('Beta', stats.ttest_rel(originalThetas[:, 1],
                                  gridThetas[:, locm, 1],
                                  nan_policy='omit'))



#%% ~~ Nelder - Mead ~~ %%#
#-------------------------#


initial_guess = np.full((len(simList), N_ITERS, 2), np.nan)
nmThetas = np.full((len(simList), len(MDLS), 2), np.nan)

start_total = time.time()
for simi, filei in enumerate(simList):
    simData = pd.read_csv(SIM_DIR / filei, index_col='Unnamed: 0')
    
    start_sim = time.time()
    for iti in range(N_ITERS):
        initial_guess[simi, iti, :] = (np.random.choice(alpha_options),
                                      np.random.choice(beta_options))
        for locm, modeli in enumerate(MDLS):
            # nmThetas[simi, locm, :] += optimize.fmin(
            #     sf.sim_negLL, initial_guess[simi, iti, :],
            #     args=(simData, modeli), ftol=0.001)
            nmThetas[simi, locm, :] += optimize.fmin(
                sf.sim_negSpearCor, initial_guess[simi, iti, :],
                args=(simData, modeli), ftol=0.001)
    print(f'Duration sim {simi}: {round((time.time() - start_sim) / 60, 2)} minutes')
print(f'Duration total: {round((time.time() - start_total) / 60, 2)} minutes')

nmThetas /= N_ITERS
# Save optimal values
for locm, modeli in enumerate(MDLS):
    title = f'sim_NelderMead_{N_ITERS}iters_softmax_{modeli}.csv'
    pd.DataFrame(nmThetas[:, locm, :], columns=MDLS).to_csv(OUT_DIR / title)


# Correlation plot
fig, axs = plt.subplots(nrows=1, ncols=2)
fig.suptitle('Parameter recovery Nelder-Mead')
for pari, ax in enumerate(axs):
    yvals = list(set(initial_guess[:, :, pari]))
    for locm, modeli in enumerate(MDLS):
        ax.plot(nmThetas[:, locm, pari], 'o', label=f'{models[modeli]}')
    ax.set_ylabel('Initial guess')
    ax.set_yticks(round(yvals, 3))

axs[0].set_xlabel('Mean recovered alpha/eta values')
axs[1].set_xlabel('Mean recovered beta values')
plt.legend()

plt.show()
plotnr += 1



#%% ~~ Accuracy ~~ %%#
######################


OptimalThetas = np.full((2, 2), np.nan)
model_acc = np.zeros((len(alpha_options), len(beta_options), len(ALL_MDLS)))

for iti in range(N_ITERS):
    for loca, alphai in enumerate(alpha_options):
        for locb, betai in enumerate(beta_options):
            for locm, modeli in enumerate(ALL_MDLS):
                model_acc[loca, locb, locm] += sf.sim_accuracy((alphai, betai),
                                                               exStruc,
                                                               model=modeli)
model_acc /= N_ITERS
# Optimal values
for locm, modeli in enumerate(MDLS):
    toploc = [i[0] for i in np.where(
        model_acc[:, :, locm] == np.min(model_acc[:, :, locm]))]
    OptimalThetas[locm, :] = [alpha_options[toploc[0]],
                              beta_options[toploc[1]]]
    
    fig, ax = plt.subplots()
    im, _ = pf.heatmap(np.rot90(model_acc[:, :, locm]), np.round(plotbetas, 3),
                np.round(alpha_options, 3), ax=ax,
                row_name='$\u03B2$', col_name='$\u03B1$',
                cbarlabel='Accuracy')
    
    plt.suptitle(f'Accuracy: Optimal values {modeli}')
    plt.show()
    plotnr += 1



# ------------------------------------------------------------------------ End
