#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Analysis behavioural fitted -- Version 4.1
Last edit:  2022/09/08
Author(s):  Geysen, Steven (SG)
Notes:      - Analysis of behavioural data after model fitting (SoftMax models)
            - Release notes:
                * Low-medium-high plot
                * Model comparison
                    - AIC
                    - BIC
                
To do:      - Model comparison
            
Questions:  
            
Comments:   AM: The data file was merged in R, from the single files generated
                in Matlab (they are also attached) - they include parameters
                from Yu & Dayan's (2005) computational model.
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
            
Sources:    List of arrays to matrix ( https://stackoverflow.com/a/48456883 )
            https://gist.github.com/jcheong0428/f25b47405d9d328691c102787bc92175#file-lmer-in-python-ipynb
            https://www.statology.org/aic-in-python/
"""



#%% ~~ Imports and directories ~~ %%#


import numpy as np
import pandas as pd

import fns.assisting_functions as af
import fns.behavioural_functions as bf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pathlib import Path
from scipy import optimize, stats

from sklearn.linear_model import LinearRegression


# Directories
SPINE = Path.cwd().parent
DATA_DIR = SPINE / 'data'
RES_DIR = SPINE / 'results'
OUT_DIR = RES_DIR / 'behavioural_analysis'
if not Path.exists(OUT_DIR):
    Path.mkdir(OUT_DIR)



#%% ~~ Variables ~~ %%#


# Uncertainty data
un_data = pd.read_csv(DATA_DIR / 'UnDataProjectSteven.csv')
## Rescale 1-2 to 0-1
scaList = [
    'relCue', 'irrelCue', 'validity', 'targetLoc', 'relCueCol', 'guessCue'
    ]
un_data.loc[:, (scaList)] = abs(un_data.loc[:, (scaList)] - 1)
## Parameter values
thetas_RW_soft = pd.read_csv(RES_DIR / 'pp_NelderMead_10iters_softmax_RW.csv',
                             index_col='Unnamed: 0')
thetas_H_soft = pd.read_csv(RES_DIR / 'pp_NelderMead_10iters_softmax_H.csv',
                            index_col='Unnamed: 0')

# Number of participants
npp = un_data['id'].max()
# Number of trials in bin
NBIN = 15
# Number of iterations
N_ITERS = 10
# Relevant variables
relVar = ['RT', 'RPE_RW', 'RPE_H', 'alpha_H']
# Models with free parameters
# MDLS = ['RW', 'H', 'M']
MDLS = ['RW', 'H']

# Plot specs
## Plot number
plotnr = 0
## Plot labels
plabels = ['All data', 'Low UUn', 'High UUn']
binlabels = ['First 15', 'Middle 15', 'Leftover', 'Last', 'Last 15']



#%% ~~ Add models ~~ %%#
########################

dataList = []

for ppi in range(npp):
    ## Skip pp6 (not in data)
    if ppi + 1 == 6:
        continue
    ## Use only data from pp
    pp_data = un_data[un_data['id'] == ppi + 1]
    pp_data.reset_index(drop=True, inplace=True)
    ## Parameter values
    thetaRW = thetas_RW_soft.loc[ppi]
    thetaH = thetas_H_soft.loc[ppi]
    
    # Add models' estimates
    Daphne = bf.ppRW_1c(thetaRW, pp_data)
    Hugo = bf.ppHybrid_1c(thetaH, Daphne)
    Wilhelm = bf.ppWSLS(Hugo)
    Renee = bf.ppRandom(Wilhelm)
    
    dataList.append(Renee)

# One data frame
complete_data = pd.concat(dataList, ignore_index=True)



#%% ~~ Exploration ~~ %%#
#########################


# Separate data sets for convenience
low_data = complete_data[complete_data['gammaBlock'] >= 0.8]
high_data = complete_data[complete_data['gammaBlock'] < 0.8]

dataList = [complete_data, low_data, high_data]

for datai, labeli in zip(dataList, plabels):
    print(labeli.center(20, '='))
    post15, middle15, leftover, last15, pre15 = af.bin_switch(
        datai, relVar, NBIN
        )
    # List of arrays to matrix
    ##SG: Works only if input arrays have the same shape (not always the case
        # with leftover and last15).
    for bini in [post15, middle15, pre15]:
        for vari in relVar:
            bini[vari] = np.stack(bini[vari], axis=0)
    
    # Grossman et al. (2022)
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(labeli)
    for vari, ploti in zip(relVar, np.ravel(axs)):
        # Mean values
        meanplot = np.ravel(np.stack([np.nanmean(pre15[vari], axis=0),
                                      np.nanmean(post15[vari], axis=0)],
                                     axis=0))
        # Standard deviation
        standivs = np.ravel(np.stack([np.nanstd(pre15[vari], axis=0),
                                      np.nanstd(post15[vari], axis=0)], axis=0))
        topsd = np.add(meanplot, standivs)
        minsd = np.subtract(meanplot, standivs)
        
        print('Paired t test', labeli, vari,
              stats.ttest_rel(np.nanmean(pre15[vari], axis=0),
                                        np.nanmean(post15[vari], axis=0),
                                        nan_policy='omit'))
        
        ## Length of switch bar depends on values
        barlen = (np.nanmin(minsd) - (np.nanmin(minsd) * 0.1),
                  np.nanmax(topsd) + (np.nanmax(topsd) * 0.1))
        ploti.vlines(15, barlen[0], barlen[1], colors='black')
        
        ploti.plot(meanplot)
        ploti.fill_between(np.linspace(0,29,30), topsd, minsd, alpha=0.2)
        
        ploti.set_xticks(np.linspace(0, 30, 5),
                         labels=[-15, 'before', 'switch', 'after', 15])
        ploti.set_xlabel('trials')
        ploti.set_ylabel(vari)
    
    plt.show()
    plotnr += 1
    
    # Marzecová et al. (2019)
    binlist = [post15, middle15, leftover, last15, pre15]
    fig, axs = plt.subplots(nrows=2, ncols=2)
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
        print('ANOVA', labeli, vari,
              stats.f_oneway(
                  plotbins[0],
                  plotbins[1],
                  # plotbins[2],
                  plotbins[3]
                  # plotbins[4]
                  )
              )
        print('*** ANOVA last categories ***', labeli, vari,
              stats.f_oneway(
                  plotbins[2],
                  plotbins[3],
                  plotbins[4]
                  )
            )
        # Plot
        # ----
        if vari == 'RT':
            ##SG: RT has too much outliers to be clear as violin plot.
            ploti.boxplot(plotbins, showfliers=False)
        else:
            ploti.violinplot(plotbins, showmeans=False, showmedians=True)
        ploti.set_xticks(
            [y + 1 for y in range(len(binlist))],
            labels=binlabels
            )
        ploti.set_ylabel(vari)
    plt.show()
    plotnr += 1



#%% ~~ Model comparison ~~ %%#
##############################


RW_predictors = ['RPE_RW']
hybrid_predictors = ['alpha_H', 'RPE_H']
general_predictors = ['trial', 'targetLoc', 'correct']

outcome_data = complete_data['RT']
lm_data_RW = complete_data[general_predictors + RW_predictors]
lm_data_RW = sm.add_constant(lm_data_RW)
lm_data_hyb = complete_data[general_predictors + hybrid_predictors]
lm_data_hyb = sm.add_constant(lm_data_hyb)

mdl_RW = sm.OLS(outcome_data, lm_data_RW, missing='drop').fit()
print(mdl_RW.summary())
print('AIC RW', mdl_RW.aic)
print('BIC RW', mdl_RW.bic)

mdl_hyb = sm.OLS(outcome_data, lm_data_hyb, missing='drop').fit()
print(mdl_hyb.summary())
print('AIC Hybrid', mdl_hyb.aic)
print('BIC Hybrid', mdl_hyb.bic)



#%% ~~ RT-PE correlations ~~ %%#
################################


# Random parameter values
pre_cors = np.zeros((npp, len(MDLS)))
for _ in range(N_ITERS):
    for ppi in range(npp):
        ## Skip pp6 (not in data)
        if ppi + 1 == 6:
            continue
        ## Use only data from pp
        pp_data = un_data[un_data['id'] == ppi + 1]
        pp_data.reset_index(drop=True, inplace=True)
        ## Parameter values
        thetas = [np.random.uniform(0, 1), np.random.uniform(0, 20)]
        
        # Add models' estimates
        for loci, modeli in enumerate(MDLS):
            pre_cors[ppi, loci] += bf.pp_negSpearCor(thetas, pp_data,
                                                     model=modeli)
pre_cors /= N_ITERS
print('mean pre cor Daphne:', np.nanmean(pre_cors[:, 0]))
print('mean pre cor Hugo:', np.nanmean(pre_cors[:, 1]))


# 'Optimised' parameter values
pp_cors = np.full((npp, len(MDLS)), np.nan)

for ppi in range(npp):
    ## Skip pp6 (not in data)
    if ppi + 1 == 6:
        continue
    ## Use only data from pp
    pp_data = un_data[un_data['id'] == ppi + 1]
    pp_data.reset_index(drop=True, inplace=True)
    ## Parameter values
    thetas = [thetas_RW_soft.loc[ppi], thetas_H_soft.loc[ppi]]
    
    # Add models' estimates
    for loci, modeli in enumerate(MDLS):
        pp_cors[ppi, loci] = bf.pp_negSpearCor(thetas[loci], pp_data,
                                               model=modeli)

print('mean cor Daphne:', np.nanmean(pp_cors[:, 0]))
print('mean cor Hugo:', np.nanmean(pp_cors[:, 1]))



# ------------------------------------------------------------------------ End
