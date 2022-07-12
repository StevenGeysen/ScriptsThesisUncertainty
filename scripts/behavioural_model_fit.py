#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Behavioural model fit -- Version 1.2
Last edit:  2022/07/12
Author(s):  Geysen, Steven (SG)
Notes:      - Fit models to behavioural data of Marzecova et al. (2019)
            - Release notes:
                * RT test
                * Start Nelder-Mead optimisation
                
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
"""



#%% ~~ Imports and directories ~~ %%#


import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import fns.plot_functions as pf
import fns.behavioural_functions as bf

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

# Number of participants
npp = un_data['id'].max()
# Plot number
plotnr = 0

# Alpha/eta options
alpha_options = np.linspace(0.01, 1, 40)
# Beta options
beta_options = np.linspace(0.1, 20, 40)



#%% ~~ RT split ~~ %%#
######################
"""
Trying to answer "How do I know the participant's selection?"

The reasoning is that RT under the median are fast RTs, and fast RTs should be
more prevalent on congruent trials. Slow RTs are then a stand in for
incongruent. This plot is to see how well this assumption can be seen in the
behavioural data.

Does not work!
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



#%% ~~ Fitting ~~ %%#
#####################


#%% ~~ Grid search ~~ %%#
#-----------------------#


# Smallest alpha and beta values left-below
plotbetas = np.flip(beta_options)
gridThetas = np.full((npp, 2), np.nan)

start_total = time.time()
for ppi in range(npp):
    ## Skip pp6 (not in data)
    if ppi + 1 == 6:
        continue
    ## Use only data from pp
    pp_data = un_data[un_data['id'] == ppi + 1]
    
    one_totpp_log = np.zeros((len(alpha_options), len(beta_options)))
    
    
    start_pp = time.time()
    for loca, alphai in enumerate(alpha_options):
        for locb, betai in enumerate(beta_options):
            one_totpp_log[loca, locb] = bf.pp_negSpearCor((alphai, betai),
                                                          pp_data,
                                                          model='RW')
    
    # Optimal values
    maxloc=[i[0] for i in np.where(one_totpp_log == np.min(one_totpp_log))]
    gridThetas[ppi -1, :] = [alpha_options[maxloc[0]], beta_options[maxloc[1]]]
    
    print(f'Duration pp {ppi}: {round((time.time() - start_pp) / 60, 2)} minutes')
    
    if ppi % 2 == 0:
        plt.figure(plotnr)
        fig, ax = plt.subplots()
        im, _ = pf.heatmap(np.rot90(one_totpp_log), np.round(plotbetas, 3),
                    np.round(alpha_options, 3), ax=ax,
                    row_name='$\u03B2$', col_name='$\u03B1$',
                    cbarlabel='Negative Spearman Correlation')
        
        plt.suptitle(f'Grid search Negative Spearman Correlation of choice for participant {ppi}')
        plt.show()
        plotnr += 1
        
        print(gridThetas[ppi - 1])
print(f'Duration total: {round((time.time() - start_total) / 60, 2)} minutes')



#%% ~~ Nelder - Mead ~~ %%#
#-------------------------#


x0 = (0.5, 10)
nmThetas = np.full((npp, 2), np.nan)

for ppi in range(npp):
    ## Skip pp6 (not in data)
    if ppi + 1 == 6:
        continue
    ## Use only data from pp
    pp_data = un_data[un_data['id'] == ppi + 1]
    pp_data.reset_index(drop=True, inplace=True)
    
    nmThetas[ppi, :] = optimize.fmin(bf.pp_negSpearCor, x0,
                                      args = (pp_data, 'RW'),
                                      ftol = 0.001)



# ------------------------------------------------------------------------ End
