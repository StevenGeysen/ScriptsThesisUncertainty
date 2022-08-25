#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Behavioural model fit -- Version 3
Last edit:  2022/08/25
Author(s):  Geysen, Steven (SG)
Notes:      - Fit models to behavioural data of Marzecova et al. (2019)
            - Release notes:
                * SoftMax
                    - Grid search
                    - Nelder-Mead
                * Argmax
                    - Grid search
                    -Nelder-Mead
                * Correlation plots
                
To do:      - Fit models
            - Statistics
            
Questions:  
            
Comments:   AM: The data file was merged in R, from the single files generated
                in Matlab (they are also attached) - they include parameters
                from the Yu & Dayan's (2005) computational model.
                The columns contain:
                    * 'id' - participants id
                    * 'block' - block # of the task
                    * 'trial' - trial # of the task
                    * 'relCue' - direction of the relevant cue
                        (1: left / 2: right)
                    * 'irrelCue'- direction of the irrelevant cue
                        (1: left / 2: right)
                    * 'validity' - validity with respect to the relevant cue
                    * 'targetLoc' - location of the target (1: left / 2: right)
                    * 'relCueCol' - color of the relevant cue
                        (1: white / 2: black)
                    * 'gammaBlock' - the validity level within the block
                    * 'RT' - response time in ms
                    * 'correct' - if the response was correct: 1, if missed: 3
                        if incorrect button: 2 (e.g., left button instead of
                        right)
                    * 'I' - parameter I from the Yu & Dayan's (2005)
                        approximate algorithm
                    * 'guessCue' - the cue which is currently assumed to be
                        correct
                    * 'Switch' - count of trials between assumed switches of
                        the cue
                    * 'Lamda' - lamda parameter from Yu & Dayan's (2005)
                        approximate algorithm - unexpected uncertainty
                    * 'Gamma' - gamma parameter from Yu & Dayan's (2005)
                        approximate algorithm - expected uncertainty
                    * 'pMui' - probability that the current context is correct
                    * 'pMuNotI'- probability that the current  context is not
                        correct
                    * 'pe' - prediction error reflecting divergence from the
                        prediction on current trial (combines Lamda and Gamma)
                    * 'logRTmod' - log prediction error
                    * 'logRTexp" - log RT
            
Sources:    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html
            https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#nelder-mead-simplex-algorithm-method-nelder-mead
            Remove NaN from array ( https://stackoverflow.com/a/11620982 )
            https://theprogrammingexpert.com/python-remove-nan-from-list/
"""



#%% ~~ Imports and directories ~~ %%#


import time

import numpy as np
import pandas as pd

import fns.assisting_functions as af
import fns.behavioural_functions as bf
import fns.plot_functions as pf
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import optimize, stats


# Directories
SPINE = Path.cwd().parent
DATA_DIR = SPINE / 'data'
OUT_DIR = SPINE / 'results'
if not Path.exists(OUT_DIR):
    Path.mkdir(OUT_DIR)



#%% ~~ Variables ~~ %%#


# Uncertainty data
un_data = pd.read_csv(DATA_DIR / 'UnDataProjectSteven.csv')
## Rescale 1-2 to 0-1
scaList = ['relCue', 'irrelCue', 'validity', 'targetLoc', 'relCueCol']
un_data.loc[:, (scaList)] = abs(un_data.loc[:, (scaList)] - 1)
## Separate data sets for convenience
valid_data = un_data[un_data['validity'] == 0]
invalid_data = un_data[un_data['validity'] == 1]

# Number of iterations
N_ITERS = 10
# Models with optimiseable parameters
# MDLS = ['RW', 'H', 'M']
MDLS = ['RW', 'H']
# Number of participants
npp = un_data['id'].max()
# Number of trials in bin
binsize = 15

# Alpha/eta options
alpha_options = np.linspace(0.01, 1, 40)
# Beta options
##SG: The SoftMax policy needs a high beta value for the model to be accurate
    # (see simulations). Therefore it is not usefull to look at beta values
    # smaller than 10.
beta_options = np.linspace(10, 20, 40)

# Plot specs
##SG: To have the smallest alpha and beta values in the same corner (left-down)
plotbetas = np.flip(beta_options)
## Plot number
plotnr = 0
## Plot labels
plabels = ['Valid trials', 'Invalid trials']
## Model labels
models = af.labelDict()



#%% ~~ Exploration ~~ %%#
#########################


#%% ~~ RT split ~~ %%#
#--------------------#
"""
Trying to answer "Can I know the participant's selection?"

The reasoning is that RT under the median are fast RTs, and fast RTs should be
more prevalent on congruent trials. Slow RTs are then a stand in for
incongruent. This plot is to see how well this assumption can be seen in the
behavioural data.

DOES NOT WORK!
"""


# Participant loop
for ppi in range(npp):
    ## Skip pp6 (not in data)
    if ppi + 1 == 6:
        continue
    ## Use only data from pp
    pp_data = un_data[un_data['id'] == ppi + 1]
    
    # Median RT
    ##SG: Excluded NaN values when computing the result.
    median_rt = pp_data['RT'].median()
    pp_data.loc[:, ('selCue_PP')] = np.where(pp_data['RT'] < median_rt,
                                             pp_data['relCueCol'],
                                             abs(1 - pp_data['relCueCol']))
    pf.selplot(pp_data, 'pp', plotnr, pp=ppi)
    plotnr += 1



#%% ~~ Plots ~~ %%#
#-----------------#


# RT distribution plot
# --------------------
fig, ax = plt.subplots()
ax.hist(valid_data['RT'], bins = 30, alpha = 0.5, label=plabels[0])
ax.hist(invalid_data['RT'], bins = 30, alpha = 0.5, label=plabels[1])

fig.suptitle('RT distribution participants', fontsize=14)
ax.set_xlabel('Response times (s)')
ax.set_ylabel('Count')
ax.legend()

plt.show()
plotnr += 1


# RT validity effect
# ------------------
valrt = np.asarray(valid_data['RT'])
invalrt = np.asarray(invalid_data['RT'])
## Remove NaN values
rt_data = [valrt[~np.isnan(valrt)], invalrt[~np.isnan(invalrt)]]

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
fig.suptitle('Validity effect')
vplot, bplot = axs[0], axs[1]
## Violin plot
vplot.violinplot(rt_data,
                  showmeans=False,
                  showmedians=True)
vplot.set_title('Violin plot participants')
## Box plot
bplot.boxplot(rt_data)
bplot.set_title('Box plot participants')

for ax in axs:
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(rt_data))],
                  labels=plabels)
    ax.set_xlabel('Trial type')
    ax.set_ylabel('Response times')

plt.show()
plotnr += 1


# RTs over time
# -------------
first_bin = []
last_bin = []


for ppi in range(npp):
    ## Skip pp6 (not in data)
    if ppi + 1 == 6:
        continue
    ## Use only data from pp
    pp_data = un_data[un_data['id'] == ppi + 1]
    pp_data.reset_index(drop=True, inplace=True)
    pp_rt = pp_data['RT']
    
    lag_relCueCol = pp_data.relCueCol.eq(pp_data.relCueCol.shift())
    switches = np.where(lag_relCueCol == False)[0]
    
    ## Bin RT of first and last b trials
    for starti, endi in af.pairwise(switches):
        first_bin.append(np.nanmean(pp_rt[starti:starti + binsize]))
        last_bin.append(np.nanmean(pp_rt[endi - binsize:endi + 1]))
    
    ## Length of switch bars depends on values of participant
    barlen = (np.nanmin(pp_rt) + (np.nanmin(pp_rt) * 0.1),
              np.nanmax(pp_rt) + (np.nanmax(pp_rt) * 0.1))
    
    # RT curve
    plt.figure(plotnr)
    plt.suptitle(f'RT curve participant {ppi}', y=.99)
    plt.plot(pp_rt, label='RT')
    plt.vlines(switches[1:], barlen[0], barlen[1], colors='black')
    
    plt.xlabel('Trials')
    plt.ylabel('Response time (s)')
    plt.ylim(barlen[0] - 50, barlen[1] + 100)
    
    plt.legend()
    plt.tight_layout()

    plt.show()
    plotnr += 1


# Binned RT
## Remove NaN values
firstlist = [i for i in first_bin if np.isnan(i) == False]
lastlist = [i for i in last_bin if np.isnan(i) == False]
plotbins = [firstlist, lastlist]

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
fig.suptitle('Mean binned RT')
vplot, bplot = axs[0], axs[1]
## Violin plot
vplot.violinplot(plotbins,
                  showmeans=False,
                  showmedians=True)
vplot.set_title('Violin plot participants')
## Box plot
bplot.boxplot(plotbins)
bplot.set_title('Box plot participants')

for ax in axs:
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(plotbins))],
                  labels=[f'First {binsize} trials',
                          f'Last {binsize} trials'])
    ax.set_xlabel('Trial numbers')
    ax.set_ylabel('Response times')

plt.show()
plotnr += 1


# =============================================================================
# Output numbers
# =============================================================================
print('RT valid trials')
print('mean', np.nanmean(valid_data['RT']), np.nanstd(valid_data['RT']))
print('RT invalid  trials')
print('mean', np.nanmean(invalid_data['RT']), np.nanstd(invalid_data['RT']))

print(f'First {binsize} trials')
print('mean', np.nanmean(first_bin), np.nanstd(first_bin))
print(f'Last {binsize} trials')
print('mean', np.nanmean(last_bin), np.nanstd(last_bin))
## Quick check if different enough
print('Paired t-test')
print(stats.ttest_rel(first_bin, last_bin, nan_policy='omit'))



#%% ~~ Fitting ~~ %%#
#####################


#%% ~~ SoftMax ~~ %%#
#-------------------#


#%% ~~ Grid search ~~ %%#


gridThetas = np.full((npp, len(MDLS), 2), np.nan)

start_total = time.time()
for ppi in range(npp):
    ## Skip pp6 (not in data)
    if ppi + 1 == 6:
        continue
    ## Use only data from pp
    pp_data = un_data[un_data['id'] == ppi + 1]
    one_totpp = np.zeros((len(alpha_options), len(beta_options), len(MDLS)))
    
    start_pp = time.time()
    for locm, modeli in enumerate(MDLS):
        for loca, alphai in enumerate(alpha_options):
            for locb, betai in enumerate(beta_options):
                for iti in range(N_ITERS):
                    one_totpp[loca, locb, locm] += bf.pp_negSpearCor(
                        (alphai, betai), pp_data, model=modeli, asm='soft'
                        )
        modeldata = one_totpp[:, :, locm] / N_ITERS
        # Optimal values
        toploc=[i[0] for i in np.where(one_totpp == np.min(one_totpp))]
        gridThetas[ppi, locm, :] = [alpha_options[toploc[0]], beta_options[toploc[1]]]
    print(f'Duration pp {ppi}: {round((time.time() - start_pp) / 60, 2)} minutes')
    
    if ppi % 2 == 0:
        plt.figure(plotnr)
        fig, ax = plt.subplots()
        im, _ = pf.heatmap(np.rot90(one_totpp), np.round(plotbetas, 3),
                    np.round(alpha_options, 3), ax=ax,
                    row_name='$\u03B2$', col_name='$\u03B1$',
                    cbarlabel='Negative Spearman Correlation')
        
        plt.suptitle(f'Grid search Negative Spearman Correlation of choice {ppi}')
        plt.show()
        plotnr += 1
        
        print(gridThetas[ppi - 1])
print(f'Duration total: {round((time.time() - start_total) / 60, 2)} minutes')

# Save optimal values
##SG: Remove empty row of participant 6
# gridThetas = np.delete(gridThetas, 5, 0)
for locm, modeli in enumerate(MDLS):
    title = f'pp_gridsearch_{N_ITERS}iters_softmax_{modeli}.csv'
    pd.DataFrame(gridThetas[:, locm, :],
                 columns=['alpha', 'beta']).to_csv(OUT_DIR / title)

# Correlation plot
fig, axs = plt.subplots(nrows=1, ncols=2)
fig.suptitle('Parameter estimation Grid search')
for pari, ax in enumerate(axs):
    for locm, modeli in enumerate(MDLS):
        ax.plot(gridThetas[:, locm, pari], 'o', label=f'{models[modeli]}')
    ax.set_ylabel('Initial guess')

axs[0].set_xlabel('Mean estimated alpha/eta values')
axs[0].set_yticks(np.round(alpha_options, 3))
axs[1].set_xlabel('Mean estimated beta values')
axs[1].set_yticks(np.round(beta_options, 3))
plt.legend()

plt.show()
plotnr += 1



#%% ~~ Nelder - Mead ~~ %%#


initial_guess = np.full((npp, N_ITERS, 2), np.nan)
nmThetas = np.full((npp, len(MDLS), 2), np.nan)

start_total = time.time()
for ppi in range(npp):
    ## Skip pp6 (not in data)
    if ppi + 1 == 6:
        continue
    ## Use only data from pp
    pp_data = un_data[un_data['id'] == ppi + 1]
    pp_data.reset_index(drop=True, inplace=True)
    
    start_pp = time.time()
    for iti in range(N_ITERS):
        initial_guess[ppi, iti, :] = (np.random.choice(alpha_options),
                                      np.random.choice(beta_options))
        for locm, modeli in enumerate(MDLS):
            nmThetas[ppi, locm, :] += optimize.fmin(
                bf.pp_negSpearCor, initial_guess[ppi, iti, :],
                args=(pp_data, modeli), ftol=0.001)
    print(f'Duration pp {ppi}: {round((time.time() - start_pp) / 60, 2)} minutes')
print(f'Duration total: {round((time.time() - start_total) / 60, 2)} minutes')

nmThetas /= N_ITERS
# Save optimal values
# nmThetas = np.delete(gridThetas, 5, 0)
for locm, modeli in enumerate(MDLS):
    title = f'pp_NelderMead_{N_ITERS}iters_softmax_{modeli}.csv'
    pd.DataFrame(nmThetas[:, locm, :],
                 columns=['alpha', 'beta']).to_csv(OUT_DIR / title)


# Correlation plot
fig, axs = plt.subplots(nrows=1, ncols=2)
fig.suptitle('Parameter estimation Nelder-Mead')
for pari, ax in enumerate(axs):
    for locm, modeli in enumerate(MDLS):
        ax.plot(nmThetas[:, locm, pari], 'o', label=f'{models[modeli]}')
    ax.set_ylabel('Initial guess')

axs[0].set_xlabel('Mean estimated alpha/eta values')
axs[0].set_yticks(np.round(alpha_options, 3))
axs[1].set_xlabel('Mean estimated beta values')
axs[1].set_yticks(np.round(beta_options, 3))
plt.legend()

plt.show()
plotnr += 1



#%% ~~ Argmax ~~ %%#
#------------------#


##SG: +1 to store optimal values
gridThetas = np.zeros((npp, len(MDLS), len(alpha_options) + 1))

start_total = time.time()
for ppi in range(npp):
    ## Skip pp6 (not in data)
    if ppi + 1 == 6:
        continue
    ## Use only data from pp
    pp_data = un_data[un_data['id'] == ppi + 1]
    pp_data.reset_index(drop=True, inplace=True)
    
    start_pp = time.time()
    for locm, modeli in enumerate(MDLS):
        for loca, alphai in enumerate(alpha_options):
            for iti in range(N_ITERS):
                gridThetas[ppi, locm, loca] += bf.pp_negSpearCor((alphai, ),
                                                                 pp_data,
                                                                 model=modeli)
        modeldata = gridThetas[ppi, locm, :-1] / N_ITERS
        # Optimal values
        toploc = [i[0] for i in np.where(modeldata == np.min(modeldata))]
        gridThetas[ppi, locm, -1] = alpha_options[toploc[0]]
        
        # Intermittent checking
        if ppi % 2 == 0:
            pf.heatmap_1d(modeli, modeldata, toploc, alpha_options,
                          gridThetas[ppi, :, -1])
            plotnr += 1
    print(f'Duration pp {ppi}: {round((time.time() - start_pp) / 60, 2)} minutes')
    print(gridThetas[ppi, :, -1])
print(f'Duration total: {round((time.time() - start_total) / 60, 2)} minutes')
# Save optimal values
title = f'pp_gridsearch_{N_ITERS}iters_argmax.csv'
##SG: Remove empty row of participant 6
# gridThetas = np.delete(gridThetas, 5, 0)
pd.DataFrame(gridThetas[:, :, -1], columns=MDLS).to_csv(OUT_DIR / title)

##SG: Paired t-test to see if the difference between the estimated model
    # parameters is big.
print(stats.ttest_rel(gridThetas[:, 0, -1], gridThetas[:, 1, -1],
                      nan_policy='omit'))


# Correlation plot
fig, ax = plt.subplots()
fig.suptitle('Parameter estimation Grid search')
for locm, modeli in enumerate(MDLS):
    # Plot
    ax.plot(gridThetas[:, locm], 'o', label=f'{models[modeli]}')

ax.set_xlabel('Mean estimated values')
ax.set_ylabel('Initial guess')
ax.set_yticks(np.round(alpha_options, 3))
plt.legend()

plt.show()
plotnr += 1



#%% ~~ Nelder - Mead ~~ %%#


initial_guess = np.full((npp, N_ITERS), np.nan)
nmThetas = np.zeros((npp, len(MDLS)))

start_total = time.time()
for ppi in range(npp):
    ## Skip pp6 (not in data)
    if ppi + 1 == 6:
        continue
    ## Use only data from pp
    pp_data = un_data[un_data['id'] == ppi + 1]
    pp_data.reset_index(drop=True, inplace=True)
    
    start_pp = time.time()
    for iti in range(N_ITERS):
        initial_guess[ppi, iti] = np.random.choice(alpha_options)
        for locm, modeli in enumerate(MDLS):
            nmThetas[ppi, locm] += optimize.fmin(bf.pp_negSpearCor,
                                                  (initial_guess[ppi, iti], ),
                                                  args=(pp_data, modeli, 'arg'),
                                                  ftol=0.001)
    print(f'Duration pp {ppi}: {round((time.time() - start_pp) / 60, 2)} minutes')
    
    if ppi % 2 == 0:
        print(initial_guess)
        print(nmThetas[ppi])
print(f'Duration total: {round((time.time() - start_total) / 60, 2)} minutes')

nmThetas /= N_ITERS
# Save optimal values
title = f'pp_NelderMead_{N_ITERS}iters_argmax.csv'
# nmThetas = np.delete(gridThetas, 5, 0)
pd.DataFrame(nmThetas, columns=MDLS).to_csv(OUT_DIR / title)


# Correlations
yvals = np.array(list(set(initial_guess)))

fig, ax = plt.subplots()
fig.suptitle('Parameter estimation Nelder-Mead')
for locm, modeli in enumerate(MDLS):
    print(modeli)
    print(stats.ttest_rel(nmThetas[:, locm], initial_guess,
                          nan_policy='omit'))
    # Plot
    ax.plot(nmThetas[:, locm], 'o', label=f'{models[modeli]}')

ax.set_xlabel('Mean estimated values')
ax.set_ylabel('Initial guess')
ax.set_yticks(np.round(yvals, 3))
plt.legend()

plt.show()
plotnr += 1



# ------------------------------------------------------------------------ End
