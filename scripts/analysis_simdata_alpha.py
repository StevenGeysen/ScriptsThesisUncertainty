#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Analysis simulations: Alpha recovery -- Version 2.2.1
Last edit:  2022/08/25
Author(s):  Geysen, Steven (SG)
Notes:      - Analysis of the task used by Marzecova et al. (2019), simulated
                with argmax policy
            - Release notes:
                * Binned PE
                
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

import fns.assisting_functions as af
import fns.plot_functions as pf
import fns.sim_functions as sf
import matplotlib.pyplot as plt

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
# Number of trials in bin
binsize = 15

# Alpha/eta options
alpha_options = np.linspace(0.01, 1, 40)

# Plot specs
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
pf.pe_validity(MDLS, simList, SIM_DIR)
plotnr += 1

# PE over time
## PE curve
pf.pe_curve(MDLS, simList, SIM_DIR, plotnr, signed=False)
plotnr += 1
for modeli in MDLS:
    pf.pe_curve(modeli, simList, SIM_DIR, plotnr, signed=False)
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
    fig.suptitle(f'Mean binned PE {models[modeli]}, Argmax')
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
                                                     model=modeli, asm='arg')
        modeldata = one_sim[:, locm] / N_ITERS
        # Optimal values
        toploc = [i[0] for i in np.where(modeldata == np.min(modeldata))]
        gridThetas[simi, locm] = alpha_options[toploc[0]]
        # Intermittent checking
        if simi % 2 == 0:
            pf.heatmap_1d(modeli, modeldata, toploc, alpha_options,
                          gridThetas[simi])
            plotnr += 1
    print(f'Duration sim {simi}: {round((time.time() - start_sim) / 60, 2)} minutes')
    print(originalThetas[simi])
    print(gridThetas[simi])
print(f'Duration total: {round((time.time() - start_total) / 60, 2)} minutes')

# Save optimal values
title = f'sim_gridsearch_{N_ITERS}iters_argmax.csv'
pd.DataFrame(gridThetas, columns=MDLS).to_csv(OUT_DIR / title)



#%% ~~ Correlations ~~ %%#


fig, ax = plt.subplots()
fig.suptitle('Parameter recovery Grid search')
for locm, modeli in enumerate(MDLS):
    print(modeli)
    ##SG: Paired t-test to see if the difference between originalThetas and
        # gridThetas is too big.
    print(stats.ttest_rel(originalThetas[:, 0], gridThetas[:, locm],
                          nan_policy='omit'))
    
    # Plot
    ax.plot(originalThetas[:, 0], gridThetas[:, locm], 'o',
            label=f'{models[modeli]}')
ax.set_xlabel('True values')
ax.set_ylabel('Recovered values')
plt.legend()

plt.show()
plotnr += 1



#%% ~~ Nelder - Mead ~~ %%#
#-------------------------#


initial_guess = np.full((len(simList), N_ITERS), np.nan)
nmThetas = np.zeros((len(simList), len(MDLS)))

for simi, filei in enumerate(simList):
    simData = pd.read_csv(SIM_DIR / filei, index_col='Unnamed: 0')
    for iti in range(N_ITERS):
        initial_guess[simi, iti] = np.random.choice(alpha_options)
        for locm, modeli in enumerate(MDLS):
            nmThetas[simi, locm] += optimize.fmin(sf.sim_negSpearCor,
                                                  (initial_guess[simi, iti], ),
                                                  args=(simData, modeli, 'arg'),
                                                  ftol=0.001)
    if simi % 2 == 0:
        print(originalThetas[simi])
        print(initial_guess[simi, iti])
        print(nmThetas[simi])

nmThetas /= N_ITERS
# Save optimal values
title = f'sim_NelderMead_{N_ITERS}iters_argmax.csv'
pd.DataFrame(nmThetas, columns=MDLS).to_csv(OUT_DIR / title)



#%% ~~ Correlations ~~ %%#
yvals = np.array(list(set(initial_guess)))

fig, ax = plt.subplots()
fig.suptitle('Parameter recovery Nelder-Mead')
for locm, modeli in enumerate(MDLS):
    print(modeli)
    print(stats.ttest_rel(nmThetas[:, locm], initial_guess,
                          nan_policy='omit'))
    # Plot
    ax.plot(nmThetas[:, locm], 'o', label=f'{models[modeli]}')

ax.set_xlabel('Mean recovered values')
ax.set_ylabel('Initial guess')
ax.set_yticks(np.round(yvals, 3))
plt.legend()

plt.show()
plotnr += 1



# ------------------------------------------------------------------------ End
