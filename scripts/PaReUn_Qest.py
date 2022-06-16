#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Parameter optimisation UN data -- Qest
Last edit:  2021/08/13
Author(s):  Geysen, Steven (01611639; SG)
Notes:      - Based on Verguts & Calderon
            - Data from Marzecova et al. (2019)
            - 4 spaces for tab
            - Release notes:
                * All written out
                * Store everything in dataframe
                    Different dataframes with RS pre and post optimisation.
                * Parameters
                    - [0] model parameter
                    - [1] SoftMax parameter
                * 1 cue updated
                * Correlation functions
                    - Estimeted cue value of selected cue
                        negative and positive
                    - Estimated prediction error of selected cue in
                        PaReUnData
                            Negative and negative with UPE
                * Loop over different values for alpha and beta
            
Comments:   SG: Check directory and file name of model script. This script
                should be in the same folder as the model script and 
                the folder 'ExperimentalData' (from GitHub).
            SG: Uses numpy matrix without titles.
            SG: Adjust settings and check if correct model is used.
            
Sources:    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html
            https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#nelder-mead-simplex-algorithm-method-nelder-mead
"""



#%% ~~ Imports ~~ %%#


import time
import NewDataModel as dm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize, stats



#%% ~~ Load data ~~ %%#


# cwd = os.getcwd()
# os.chdir(cwd + '\ExperimentalData')
# In arrays
un_data = np.genfromtxt('UnDataProjectSteven.csv', delimiter = ',')
## Remove titles (or find a way that it is not NaN)
un_data = un_data[1:, 1:]

# The data file was merged in R, from the single files generated in Matlab (they are also attached) - they include parameters from the Yu & Dayan's (2005) computational model.
# 
# The columns contain:
# 0. 'id' - participants id
# 1. 'block' - block # of the task
# 2. 'trial' - trial # of the task
# 3. 'relCue' - direction of the relevant cue (1: left / 2: right)
# 4. 'irrelCue'- direction of the irrelevant cue (1: left / 2: right)
# 5. 'validity' - validity with respect to the relevant cue
# 6. 'targetLoc' - location of the target (1: left / 2: right)
# 7. 'relCueCol' - color of the relevant cue (1: white / 2: black)
# 8. 'gammaBlock' - the validity level within the block
# 9. 'RT' - response time in ms
# 10. 'correct' - if the response was correct: 1, if missed: 3 if incorrect button: 2 (e.g., left button instead of right)
# 11. 'I' - parameter I from the Yu & Dayan's (2005) approximate algorithm
# 12. 'guessCue' - the cue which is currently assumed to be correct
# 13. 'Switch' - count of trials between assumed switches of the cue
# 14. 'Lamda' - lamda parameter from Yu & Dayan's (2005) approximate algorithm - unexpected uncertainty
# 15. 'Gamma' - gamma parameter from Yu & Dayan's (2005) approximate algorithm - expected uncertainty
# 16. 'pMui' - probability that the current context is correct
# 17. 'pMuNotI'- probability that the current  context is not correct
# 18. 'pe' - prediction error reflecting divergence from the prediction on current trial (combines Lamda and Gamma)
# 19. 'logRTmod' - log prediction error
# 20. 'logRTexp" - log RT



#%% ~~ Settings ~~ %%#

## Set plot number
plotnr = 0
## Set number of participants to save time
# npp = 5
npp = int(np.amax(un_data[:, 0]))  # All participants
## Set number of iterations
iters = 1
## Start parameters
x0 = np.array([0.5, 15])
## Alpha options
alphaOptions = np.array([0.2, 0.5, 0.7])
## Beta options
betaOptions = np.array([0.5, 1, 2, 15])
## Model
opt_model = 'Hybrid'



#%% ~~ Functions ~~ %%#
#######################


def SpearCorCue(parameters=(0.01, 0.5), model='RW', data=un_data):
    """
    Negative Spearman correlation of learning model for reaction time and
        estimated value of selected cue
    NaN omited.
    Fast models.
    Non-absolute values.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is the learning rate in the model.
        Second parameter is the constant in the action selection method.
        The default is (0.01, 0.5).
    model : string
        Learning model
            * 'rw' for Rescorla-Wagner
            * 'hybrid' for RW-PH hybrid model.
        The default is 'RW'.
    data : array
        DESCRIPTION. The default is un_data.

    Returns
    -------
    negative spearman r

    """


    # Estimated values
    #-----------------
    if model.upper() == 'RW':
        ## parameters[0] = alpha, relCue = 3, irrelCue = 4, relCueCol = 7, targetLoc = 6, data
        Q_est, selcues, pe = dm.frw_1c(parameters = parameters, data = data)
        
    elif model.upper() == 'HYBRID':
        ## parameters[0] = eta, relCue = 3, irrelCue = 4, relCueCol = 7, targetLoc = 6, data, salpha = 0.01
        Q_est, selcues, pe = dm.fhybrid_1c(parameters = parameters, data = data)

    ## Estimated value of selected cue
    EstSel = np.full((data.shape[0], 1), np.nan)
    ## Prediction error of selected cue
    peSel = np.full((data.shape[0], 1), np.nan)
    for trial in range(data.shape[0]):
        EstSel[trial] = Q_est[trial, selcues[trial]]
        peSel[trial] = pe[trial]
    ## Reaction times
    # RT = data[:, 9].reshape((-1, 1))
    ## Response speed
    RS = data[:, 20].reshape((-1, 1))

    # Correlation between RT and PE
    # return - stats.spearmanr(RT, EstSel, nan_policy = 'omit')[0]
    # Correlation between RS and PE
    return - stats.spearmanr(RS, EstSel, nan_policy = 'omit')[0]


def SpearCorCuePos(parameters=(0.01, 0.5), model='RW', data=un_data):
    """
    Negative Spearman correlation of learning model for reaction time and
        estimated value of selected cue
    NaN omited.
    Fast models.
    Non-absolute values.
    No '-' before correlation.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is the learning rate in the model.
        Second parameter is the constant in the action selection method.
        The default is (0.01, 0.5).
    model : string
        Learning model
            * 'rw' for Rescorla-Wagner
            * 'hybrid' for RW-PH hybrid model.
        The default is 'RW'.
    data : array
        DESCRIPTION. The default is un_data.

    Returns
    -------
    negative spearman r

    """


    # Estimated values
    #-----------------
    if model.upper() == 'RW':
        ## parameters[0] = alpha, relCue = 3, irrelCue = 4, relCueCol = 7, targetLoc = 6, data
        Q_est, selcues, pe = dm.frw_1c(parameters = parameters, data = data)
        
    elif model.upper() == 'HYBRID':
        ## parameters[0] = eta, relCue = 3, irrelCue = 4, relCueCol = 7, targetLoc = 6, data, salpha = 0.01
        Q_est, selcues, pe = dm.fhybrid_1c(parameters = parameters, data = data)

    ## Estimated value of selected cue
    EstSel = np.full((data.shape[0], 1), np.nan)
    ## Prediction error of selected cue
    peSel = np.full((data.shape[0], 1), np.nan)
    for trial in range(data.shape[0]):
        EstSel[trial] = Q_est[trial, selcues[trial]]
        peSel[trial] = pe[trial]
    ## Reaction times
    # RT = data[:, 9].reshape((-1, 1))
    ## Response speed
    RS = data[:, 20].reshape((-1, 1))

    # Correlation between RT and PE
    # return stats.spearmanr(RT, EstSel, nan_policy = 'omit')[0]
    # Correlation between RS and PE
    return stats.spearmanr(RS, EstSel, nan_policy = 'omit')[0]



#%% ~~ Dataframe ~~ %%#


column_list = ['participant', 'guessAlpha', 'guessBeta',
               'cors_pre', 'p_pre',
               'opt_alpha_neg', 'opt_beta_neg', 'opt_alpha_pos', 'opt_beta_pos',
               'cors_post_neg', 'p_post_neg', 'cors_post_pos', 'p_post_pos']

data = pd.DataFrame(columns=column_list)


column_list_pre = ['participant', 'guessAlpha', 'guessBeta', 'RS',
                   'Qest_pre', 'selcue_pre', 'pe_pre']
datapre = pd.DataFrame(columns=column_list_pre)

column_list_post = ['participant', 'guessAlpha', 'guessBeta', 'RS',
                    'Qest_post_neg', 'selcue_post_neg', 'pe_post_neg',
                    'Qest_post_pos', 'selcue_post_pos', 'pe_post_pos']
datapost = pd.DataFrame(columns=column_list_post)



#%% ~~ Optimisation ~~ %%#
##########################


indexnr = 0
easypre = 0
easypost = 0

cors_pre = np.zeros(2)
# Negative correlation
cors_post_neg = np.zeros(2)
# Positive correlation
cors_post_pos = np.zeros(2)



start_total = time.time()
# Alpha loop
for alpha in range(len(alphaOptions)):
    x0[0] = alphaOptions[alpha]
    start_alpha = time.time()
    # Beta loop
    for beta in range(len(betaOptions)):
        x0[1] = betaOptions[beta]
        start_beta = time.time()
        # Participant loop
        for pp in range(npp):
            ## Skip pp6 (not in data)
            if pp + 1 == 6:
                continue
            ## Use only data from pp
            pp_data = un_data[un_data[:, 0] == pp + 1]
            print(pp, beta, alpha)


            # Pre optimization
            ##################
            if opt_model.upper() == 'RW':
                Qest_pre, selcue_pre, pe_pre = dm.frw_1c(parameters = x0, data = pp_data)
            elif opt_model.upper() == 'HYBRID':
                Qest_pre, selcue_pre, pe_pre = dm.fhybrid_1c(parameters = x0, data = pp_data)
            
            EstSel = np.full((pp_data.shape[0], 1), np.nan)
            for triali in range(pp_data.shape[0]):
                ## Estimated value of selected cue
                EstSel[triali] = Qest_pre[triali, selcue_pre[triali]]
                ## Store dataframe pre optimisation
                datapre.loc[easypre] = [pp, alphaOptions[alpha], betaOptions[beta], pp_data[triali, 20],
                                       Qest_pre[triali, selcue_pre[triali]], selcue_pre[triali], pe_pre[triali]]
                easypre += 1
            
            
            # if pp < 5:
            #     # Plot
            #     # Plot number
            #     plt.figure(plotnr)
            #     plt.title(f'Cue selection / Qest / pp {pp} / alpha {alphaOptions[alpha]} / beta {betaOptions[beta]}')
            #         ## Cue estimates
            #     plt.plot(Qest_pre[:, 0], label = 'Cue 1')
            #     plt.plot(Qest_pre[:, 1], label = 'Cue 2')
            #         ## Selected cue
            #     plt.plot(selcue_pre[:], label = 'SelCue')
            #         ## True cue
            #     plt.plot(pp_data[:,  7] -1, label = 'True cue', linestyle = '-.')
            #         ## Legend
            #     plt.legend()
                
            #     plt.show()
            #     plotnr += 1
            
            # Correlation PE-RT
            #------------------
            ## Reaction times
            # RT = pp_data[:, 9].reshape((-1, 1))
            # cors_pre = stats.spearmanr(RT, EstSel, nan_policy = 'omit')
            ## Response speed
            RS = pp_data[:, 20].reshape((-1, 1))
            cors_pre = stats.spearmanr(RS, EstSel, nan_policy = 'omit')


            # Optimization
            ##############
            ## Negative correlation
            estPar_neg = optimize.fmin(SpearCorCue, x0, args = (opt_model, pp_data), ftol = 0.001)
            ## Positive correlation
            estPar_pos = optimize.fmin(SpearCorCuePos, x0, args = (opt_model, pp_data), ftol = 0.001)


            # Post optimization
            ###################
            if opt_model.upper() == 'RW':
                # Negative correlation
                Qest_post_neg, selcue_post_neg, pe_post_neg = dm.frw_1c(parameters = estPar_neg, data = pp_data)
                # Positive correlation
                Qest_post_pos, selcue_post_pos, pe_post_pos = dm.frw_1c(parameters = estPar_pos, data = pp_data)
            elif opt_model.upper() == 'HYBRID':
                # Negative correlation
                Qest_post_neg, selcue_post_neg, pe_post_neg = dm.fhybrid_1c(parameters = estPar_neg, data = pp_data)
                # Positive correlation
                Qest_post_pos, selcue_post_pos, pe_post_pos = dm.fhybrid_1c(parameters = estPar_pos, data = pp_data)
            
            EstSel_neg = np.full((pp_data.shape[0], 1), np.nan)
            EstSel_pos = np.full((pp_data.shape[0], 1), np.nan)
            for triali in range(pp_data.shape[0]):
                EstSel_neg[triali] = Qest_post_neg[triali, selcue_post_neg[triali]]
                EstSel_pos[triali] = Qest_post_pos[triali, selcue_post_pos[triali]]
                ## Store dataframe post optimisation
                datapost.loc[easypost] = [pp, alphaOptions[alpha], betaOptions[beta], pp_data[triali, 20],
                                        Qest_post_neg[triali, selcue_post_neg[triali]], selcue_post_neg[triali], pe_post_neg[triali],
                                        Qest_post_pos[triali, selcue_post_pos[triali]], selcue_post_pos[triali], pe_post_pos[triali]]
                easypost += 1
            
            
            # if pp < 5:
            # # Selection plot
            #     plt.figure(plotnr)
            #     plt.subplot(2, 1, 1)
            #     plt.title(f'Cue selection {pp} post')
            #     plt.xlabel('Negative correlation')
            #         ## Cue estimates
            #     plt.plot(Qest_post_neg[:, 0], label = 'Cue 1')
            #     plt.plot(Qest_post_neg[:, 1], label = 'Cue 2')
            #         ## Selected cue
            #     plt.plot(selcue_post_neg[:], label = 'selCue')
            #         ## True cue
            #     plt.plot(pp_data[:, 7] -1, label = 'True cue', linestyle = '-.')
            #         ## Legend
            #     plt.legend()
                
            #     plt.subplot(2, 1, 2)
            #     plt.xlabel('Positive correlation')
            #         ## Cue estimates
            #     plt.plot(Qest_post_pos[:, 0], label = 'Cue 1')
            #     plt.plot(Qest_post_pos[:, 1], label = 'Cue 2')
            #         ## Selected cue
            #     plt.plot(selcue_post_pos[:], label = 'selCue')
            #         ## True cue
            #     plt.plot(pp_data[:, 7] -1, label = 'True cue', linestyle = '-.')
            #         ## Legend
            #     plt.legend()
                
            #     plt.show()
            #     plotnr += 1
            
            # Correlation PE-RT
            #------------------
            ## Reaction times
            # RT = pp_data[:, 9].reshape((-1, 1))
            # cors_post_na = stats.spearmanr(RT, EstSel_na, nan_policy = 'omit')
            # cors_post_pos = stats.spearmanr(RT, EstSel_pos, nan_policy = 'omit')
            ## Response speed
            RS = pp_data[:, 20].reshape((-1, 1))
            cors_post_neg = stats.spearmanr(RS, EstSel_neg, nan_policy = 'omit')
            cors_post_pos = stats.spearmanr(RS, EstSel_pos, nan_policy = 'omit')
            
            # if pp < 5:
            #     # Plot
            #     plt.figure(plotnr)
            #     plt.subplot(2, 1, 1)
            #     plt.title(f'Correlation: 1 cue / Pre-Post / {pp} / pe')
            #     plt.xlabel('Negative correlation')
            #     plt.scatter(RS, EstSel, alpha = 0.5, label = 'Pre')
            #     plt.scatter(RS, EstSel_neg, alpha = 0.5, label = 'Post')
            #     plt.legend()
                
            #     plt.subplot(2, 1, 2)
            #     plt.xlabel('Positive correlation')
            #     plt.scatter(RS, EstSel, alpha = 0.5, label = 'Pre')
            #     plt.scatter(RS, EstSel_pos, alpha = 0.5, label = 'Post')
                
            #     plt.show()
            #     plotnr += 1


            ## Store dataframe
            data.loc[indexnr] = [pp, alphaOptions[alpha], betaOptions[beta],
                                 cors_pre[0], cors_pre[1],
                                 estPar_neg[0], estPar_neg[1], estPar_pos[0], estPar_pos[1],
                                 cors_post_neg[0], cors_post_neg[1], cors_post_pos[0], cors_post_pos[1]]
            indexnr += 1


        end_beta = time.time()
        print(f'Duration beta {beta}: {round((end_beta - start_beta) / 60, 2)} minutes')
    end_alpha = time.time()
    print(f'Duration alpha {alpha}: {round((end_alpha - start_alpha) / 60, 2)} minutes')
end_total = time.time()
print(f'Total duration: {round((end_total - start_total) / 60, 2)} minutes')


data.to_csv(f'Qestreal_{opt_model}.csv', index = False)
datapre.to_csv(f'Qestreal_{opt_model}_pre.csv', index = False)
datapost.to_csv(f'Qestreal_{opt_model}_post.csv', index = False)



#------------------------------------------------------------------------- End
