#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Analysis optimal SoftMax -- Version 2.1
Last edit:  2022/09/13
Author(s):  Geysen, Steven (SG)
Notes:      - Analysis of data simulated with optimal parameter values
                according to grid search(SoftMax models)
            - Release notes:
                * Validity effect
                
To do:      - Model comparison
            
Questions:  
            
Comments:   
            
Sources:    List of arrays to matrix ( https://stackoverflow.com/a/48456883 )
            https://gist.github.com/jcheong0428/f25b47405d9d328691c102787bc92175#file-lmer-in-python-ipynb
            https://www.statology.org/aic-in-python/
            https://www.pythonfordatascience.org/anova-python/
"""



#%% ~~ Imports and directories ~~ %%#


import numpy as np
import pandas as pd

import fns.assisting_functions as af
import fns.plot_functions as pf
import fns.sim_functions as sf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pathlib import Path
from scipy import optimize, stats

from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols


# Directories
SPINE = Path.cwd().parent
RES_DIR = SPINE / 'results'
SIM_DIR = RES_DIR / 'simulations'
SOFT_DIR = SIM_DIR / 'softmax'
DATA_DIR = SOFT_DIR / 'optimal'
OUT_DIR = RES_DIR / 'analysis_optimal_soft'
if not Path.exists(OUT_DIR):
    Path.mkdir(OUT_DIR)



#%% ~~ Variables ~~ %%#


# Filenames of simulated data
simList = [filei.name for filei in Path.iterdir(DATA_DIR)]
# simList = simList[:20]  ##SG: First few to test everything quickly.
## One data frame
dataList = []
for filei in simList:
    dataList.append(pd.read_csv(DATA_DIR / filei, index_col='Unnamed: 0'))
all_data = pd.concat(dataList, ignore_index=True)

# Experimental structure
exStruc = pd.read_csv(DATA_DIR / simList[0], index_col='Unnamed: 0')
# Parameter values
thetas = np.genfromtxt(SIM_DIR / 'sim_gridsearch_50iters_softmax_optimal.csv',
                       skip_header=1, delimiter=',')[:, 1:]

# Number of trials in bin
NBIN = 15
# Number of iterations
N_ITERS = 10
# Number of simulations
nsims = len(simList)
# Relevant variables
relVar = ['rt_RW', 'RPE_RW', 'rt_H', 'RPE_H', 'alpha_H']
# Models with free parameters
MDLS = ['RW', 'H']

# Plot specs
## Plot number
plotnr = 0
## Plot labels
plabels = ['All', 'Low', 'High']
binlabels = ['First 15', 'Middle 15', 'Last']



#%% ~~ Exploration ~~ %%#
#########################


#%% ~~ Plots ~~ %%#
#-----------------#


# Learning curve
pf.learning_curve(simList, DATA_DIR, plotnr)
plotnr += 1
for modeli in ['RW', 'H', 'W', 'R']:
    print(sum(all_data['relCue'] == all_data[f'selCue_{modeli}']) / len(all_data))

# Stay behaviour
pf.p_stay(simList, DATA_DIR, plotnr)
plotnr += 1



#%% ~~ Validity effect ~~ %%#
#---------------------------#


for labeli in plabels:
    print(f' {labeli} UUn '.center(20, '='))
    post15, middle15, _, last15, _ = sf.var_bin_switch(
        simList, DATA_DIR, relVar, NBIN, labeli
        )
    
    # Marzecová et al. (2019)
    binlist = [post15, middle15, last15]
    fig, axs = plt.subplots(nrows=2, ncols=3)
    fig.suptitle(labeli)
    for vari, ploti in zip(relVar, np.ravel(axs)):
        # Remove NaN
        plotbins = []
        for bini in binlist:
            data_1d = np.concatenate(bini[vari], axis=None)
            filtered_data = data_1d[~np.isnan(data_1d)]
            plotbins.append(filtered_data)
        # ANOVA
        # -----
        print('ANOVA', vari,
              stats.f_oneway(
                  plotbins[0],
                  plotbins[1],
                  plotbins[2]
                  )
              )
        # Plot
        # ----
        # ploti.boxplot(plotbins)
        ploti.violinplot(plotbins, showmeans=False, showmedians=True)
        ploti.set_xticks(
            [y + 1 for y in range(len(binlist))],
            labels=binlabels
            )
        ploti.set_ylabel(vari)
    plt.show()
    plotnr += 1



#%% ~~ Trial position ~~ %%#


lag_relCueCol = all_data.relCueCol.eq(all_data.relCueCol.shift())
switch_points = np.where(lag_relCueCol == False)[0]
switch_points = np.append(switch_points, len(all_data))

posList = []
for starti, endi in af.pairwise(switch_points):
    
    nover = endi - starti - (2 * NBIN)
    firstdata = all_data.loc[starti:(starti + NBIN - 1)]
    firstdata['trial_pos'] = 'first'
    firstdata.reset_index(drop=True, inplace=True)
    
    middledata = all_data.loc[(starti + NBIN):(starti + 2 * NBIN - 1)]
    middledata['trial_pos'] = 'middle'
    middledata.reset_index(drop=True, inplace=True)
    
    # temdata = pd.concat([firstdata, middledata], axis=1)
    
    if nover >= NBIN:
        lastdata = all_data.loc[(endi - NBIN):(endi - 1)]
    else:
        lastdata = all_data.loc[(endi - nover):(endi - 1)]
    lastdata['trial_pos'] = 'last'
    lastdata.reset_index(drop=True, inplace=True)
    
    posList.append(pd.concat([firstdata, middledata, lastdata]))
posdata = pd.concat(posList)
posdata = posdata.reset_index(drop=True)



#%%

outcome_vars = ['rt_RW', 'RPE_RW', 'rt_H', 'RPE_H']
for outcomei in outcome_vars:
    print(f' {outcomei} '.center(20, '='))
    Lmdl = ols(
        f'{outcomei} ~ validity * gammaBlock * C(trial_pos, Treatment)',
        data=posdata
        ).fit()
    # print(Lmdl.summary())
    table = sm.stats.anova_lm(Lmdl, typ=3)
    print(table)

detailmdl = ols(
    'rt_H ~ validity * gammaBlock * C(trial_pos, Treatment)',
    data=posdata
    ).fit()
datailsum = detailmdl.summary()
# print(datailsum)
detable = sm.stats.anova_lm(detailmdl, typ=3)
print(detable)



#%% ~~ RT-PE correlations ~~ %%#
################################


# 'Optimised' parameter values
simcors = np.full((nsims, len(MDLS)), np.nan)

for simi, filei in enumerate(simList):
    simdata = pd.read_csv(DATA_DIR / filei, index_col='Unnamed: 0')
    
    # Add models' estimates
    for loci, modeli in enumerate(MDLS):
        simcors[simi, loci] = sf.sim_negSpearCor(thetas[loci, :], simdata,
                                                 model=modeli)

print('mean cor Daphne:', np.nanmean(simcors[:, 0]))
print('mean cor Hugo:', np.nanmean(simcors[:, 1]))


# ------------------------------------------------------------------------ End
