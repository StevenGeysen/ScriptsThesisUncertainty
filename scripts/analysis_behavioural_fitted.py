#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Analysis behavioural fitted -- Version 2
Last edit:  2022/09/01
Author(s):  Geysen, Steven (SG)
Notes:      - Analysis of behavioural data after model fitting
            - Release notes:
                * Plot before-after switch (as in Grossman et al., 2022)
                
To do:      - Plots expected uncertainty
                * Low-medium-high (Marzecova et al., 2019)
            - Model comparison
                * LME
            - Clean bin-spaghetti
            
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
"""



#%% ~~ Imports and directories ~~ %%#


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

# Number of iterations
N_ITERS = 10
# Models with optimiseable parameters
# MDLS = ['RW', 'H', 'M']
MDLS = ['RW', 'H']
# Number of participants
npp = un_data['id'].max()
# Number of trials in bin
NBIN = 15

# Alpha/eta options
alpha_options = np.linspace(0.01, 1, 20)
# Beta options
##SG: The SoftMax policy needs a high beta value for the model to be accurate
    # (see simulations). Therefore it is not usefull to look at beta values
    # smaller than 10.
beta_options = np.linspace(10, 20, 20)

# Plot specs
## Plot number
plotnr = 0
## Plot labels
plabels = ['Valid trials', 'Invalid trials']
## Model labels
models = af.labelDict()
##SG: To have the smallest alpha and beta values in the same corner (left-down)
plotbetas = np.flip(beta_options)



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


# Trial bins
relVal = ['RT', 'RPE_RW', 'RPE_H', 'alpha_H']
##SG: First 15 trials after switch.
post15 = {vali:[] for vali in relVal}
##SG: Trials 15 to 30.
middle15 = {vali:[] for vali in relVal}
##SG: All trials except for first 30.
leftover = {vali:[] for vali in relVal}
##SG: Last 15 trials, less if there were less then 45 trials between switches.
last15 = {vali:[] for vali in relVal}
##SG: The 15 trials before switch.
pre15 = {vali:[] for vali in relVal}

binlist = [post15, middle15, leftover, last15, pre15]


for ppi in range(npp):
    ## Skip pp6 (not in data)
    if ppi + 1 == 6:
        continue
    ## Use only data from pp
    pp_data = complete_data[complete_data['id'] == ppi + 1]
    pp_data.reset_index(drop=True, inplace=True)
    
    # Add information from 30 first trials
    for loci, bini in enumerate(binlist[:2]):
        bindata = pp_data.loc[(loci * NBIN):(loci * NBIN + NBIN - 1)]
        for vali in relVal:
            bini[vali].append(bindata[vali].to_numpy())
    
    # Switch points
    lag_relCueCol = pp_data.relCueCol.eq(pp_data.relCueCol.shift())
    switch_points = np.where(lag_relCueCol == False)[0][1:]
    
    # Add information of first last 15 trials
    startover = switch_points[0] - 30
    for vali in relVal:
        leftover[vali].append(
            pp_data.loc[(2 * NBIN):(switch_points[0] - 1)][vali].to_numpy()
            )
        pre15[vali].append(
            pp_data.loc[(switch_points[0] - NBIN):(switch_points[0] - 1)][vali].to_numpy()
            )
        ##SG: Last 15 trials or less if there were less than 45 trials
            # between start and first switch.
        if startover >= 15:
            last15[vali].append(
                pp_data.loc[(switch_points[0] - NBIN):(switch_points[0] - 1)][vali].to_numpy()
                )
        else:
            last15[vali].append(
                pp_data.loc[(switch_points[0] - startover):(switch_points[0] - 1)][vali].to_numpy()
                )
    
    # Add information of middle part
    for starti, endi in af.pairwise(switch_points):
        print(starti, endi)
        nover = endi - starti - 30
        for vali in relVal:
            post15[vali].append(
                pp_data.loc[starti:(starti + NBIN - 1)][vali].to_numpy()
                )
            middle15[vali].append(
                pp_data.loc[(starti + NBIN):(starti + 2 * NBIN - 1)][vali].to_numpy()
                )
            leftover[vali].append(
                pp_data.loc[(starti + 2 * NBIN):(endi - 1)][vali].to_numpy()
                )
            pre15[vali].append(
                pp_data.loc[(endi - NBIN):(endi - 1)][vali].to_numpy()
                )
            ##SG: Last 15 trials or less if there were less than 45 trials
                # between switches.
            if nover >= 15:
                last15[vali].append(
                    pp_data.loc[(endi - NBIN):(endi - 1)][vali].to_numpy()
                    )
            else:
                last15[vali].append(
                    pp_data.loc[(endi - nover):(endi - 1)][vali].to_numpy()
                    )
    
    # Add information of last 15 trials
    endover = len(pp_data) - endi
    for vali in relVal:
        post15[vali].append(
            pp_data.loc[endi:(endi + NBIN - 1)][vali].to_numpy()
            )
        middle15[vali].append(
            pp_data.loc[(endi + NBIN):(endi + 2 * NBIN - 1)][vali].to_numpy()
            )
        leftover[vali].append(
            pp_data.loc[(endi + 2 * NBIN):][vali].to_numpy()
            )
        pre15[vali].append(pp_data.tail(NBIN)[vali].to_numpy())
        ##SG: Last 15 trials or less if there were less than 45 trials
            # between last switch and end.
        if endover >= 15:
            last15[vali].append(pp_data.tail(NBIN)[vali].to_numpy())
        else:
            last15[vali].append(pp_data.tail(endover)[vali].to_numpy())


# List of arrays to matrix
##SG: Works only if input arrays have the same shape (not always the case with
    # leftover and last15).
for bini in [post15, middle15, pre15]:
    for vali in relVal:
        bini[vali] = np.stack(bini[vali], axis=0)



#%% ~~ Plots ~~ %%#
#-----------------#


fig, axs = plt.subplots(nrows=2, ncols=2)
for vali, ploti in zip(relVal, np.ravel(axs)):
    # Mean values
    meanplot = np.ravel(np.stack([np.nanmean(pre15[vali], axis=0),
                                  np.nanmean(post15[vali], axis=0)], axis=0))
    # Standard deviation
    standivs = np.ravel(np.stack([np.nanstd(pre15[vali], axis=0),
                                  np.nanstd(post15[vali], axis=0)], axis=0))
    topsd = np.add(meanplot, standivs)
    minsd = np.subtract(meanplot, standivs)
    
    ## Length of switch bar depends on values
    barlen = (np.nanmin(minsd) - (np.nanmin(minsd) * 0.1),
              np.nanmax(topsd) + (np.nanmax(topsd) * 0.1))
    ploti.vlines(15, barlen[0], barlen[1], colors='black')
    
    ploti.plot(meanplot)
    ploti.fill_between(np.linspace(0,29,30), topsd, minsd, alpha=0.2)
    
    ploti.set_xticks(np.linspace(0, 30, 5),
                  labels=[-15, 'before', 'switch', 'after', 15])
    ploti.set_xlabel('trials')
    ploti.set_ylabel(vali)

plt.show()
plotnr += 1


# ------------------------------------------------------------------------ End
