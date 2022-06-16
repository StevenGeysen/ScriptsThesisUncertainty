#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Thesis Simulated Paremeter Retrieval
Last edit:  2021/08/13
Author(s):  Geysen, Steven (01611639; SG)
Notes:      - Based on Verguts & Calderon
            - Data from Marzecova et al. (2019)
            - 4 spaces for tab
            - Release notes:
                * Store everything in dataframe
                    Different dataframes with RS pre and post optimisation.
                * Parameters
                    - [0] model parameter
                    - [1] selection method parameter
                * 1 cue updated
                * Correlation functions
                    - Estimated value of prediction error of selected cue
                        * negative (na) and negative with absolute values (abs)
                        * positive (pos)
                    - Estimeted cue value of selected cue in SiPaRe_Qest
                * Loop over different values for alpha and beta
            
Comments:   SG: Check directory and file name of model script. This script
                should be in the same folder as the model script and
                the folder 'Simulations'.
            SG: Adjust settings
            SG: Check if correct model is commented
            
Sources:    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html
            https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#nelder-mead-simplex-algorithm-method-nelder-mead
"""



#%% ~~ Imports ~~ %%#


import os
import time
import DataModelSim as dms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats



#%% ~~ Load data ~~ %%#

# List of simulated data
simList = os.listdir(os.getcwd())
SimData = np.zeros((2, 5))

# The data files were made with SimulationScript.
# RT = stats.exponnorm.rvs(K = abs(pe[selCue]) / 0.02635, loc = 0.3009, scale = 0.02635)
# 
# The columns contain:
## 0. 'id' - participants id
## 1. 'trial' - trial # of the task
## 2. 'relCue' - direction of the relevant cue (0: left / 1: right)
## 3. 'irrelCue'- direction of the irrelevant cue (0: left / 1: right)
## 4. 'targetLoc' - location of the target (0: left / 1: right)
## 5. 'relCueCol' - colour of the relevant cue (0: white / 1: black)
## 6. 'Choice' - selected cue
## 7. 'Reward' - validity of the selected cue
## 8. 'Cue_1 est' - estimated value of left cue
## 9. 'Cue_2 est' - estimated value of right cue
## 10. 'Cue_1 pe' - prediction error reflecting divergence from the prediction on current trial for left cue
## 11. 'Cue_2 pe' - prediction error reflecting divergence from the prediction on current trial for right cue
## 12. 'RT' - response time in s
## 13. 'selPE' -- prediction error of selected cue
## 14. 'selQ_est' -- estimated cue value of selected cue
## 15. 'sumQ_est' -- sum of estimated cue values
## 16. 'PE valid' -- prediction error in valid trials
## 17. 'PE invalid' -- prediction error in invalid trials
## 18. 'RT valid' -- reaction time in valid trials
## 19. 'RT invalid' -- reaction time in invalid trials



#%% ~~ Settings ~~ %%#

## Set plot number
plotnr = 0
## Set number of files to save time
# nsim = 5
nsim = int(len(simList))  # All files
## Start parameters
x0 = np.array([0.1, 1])
## Starting values alpha
## SG: Ground truth alphas used in simulation: 0.05, 0.1, 0.25, 0.5
alphaOptions = np.array([0.2, 0.5, 0.7])
## Starting values beta
## SG: Ground truth betas used in simulation: 1, 1.5, 2, 2.5, 3, 15
betaOptions = np.array([1, 2, 3, 10, 20])
## Model
opt_model = 'Hybrid'



#%% ~~ Functions ~~ %%#
#######################


def fModelSpearCor(parameters=(0.01, 0.5), model='RW', data=SimData):
    """
    Negative Spearman correlation of learning model
    NaN omited.
    Fast models.
    Signed prediction errors.

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
        Simulated data that has to be optimised
        The default is SimData.

    Returns
    -------
    negative spearman r

    """


    # Estimated values
    #-----------------
    if model.upper() == 'RW':
        ## parameters[0] = alpha, relCue = 2, irrelCue = 3, relCueCol = 4, targetLoc = 5, data
        Q_est, selcues, pe = dms.frw_1c(parameters = parameters, data = data)
        
    elif model.upper() == 'HYBRID':
        ## parameters[0] = eta, relCue = 2, irrelCue = 3, relCueCol = 4, targetLoc = 5, data, salpha = 0.01
        Q_est, selcues, pe = dms.fhybrid_1c(parameters = parameters, data = data)

    ## Estimated value of selected cue
    EstSel = np.full((data.shape[0], 1), np.nan)
    ## Prediction error of selected cue
    peSel = np.full((data.shape[0], 1), np.nan)
    for trial in range(data.shape[0]):
        EstSel[trial] = Q_est[trial, selcues[trial]]
        peSel[trial] = pe[trial]
    ## Reaction times
    RT = data[:, 12].reshape((-1, 1))

    # Correlation between RS and PE
    return - stats.spearmanr(RT, peSel, nan_policy = 'omit')[0]


def fModelSpearCorAbs(parameters=(0.01, 0.5), model='RW', data=SimData):
    """
    Negative Spearman correlation of learning model
    NaN omited.
    Fast models.
    Unsigned prediction errors.

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
        Simulated data that has to be optimised
        The default is SimData.

    Returns
    -------
    negative spearman r

    """


    # Estimated values
    #-----------------
    if model.upper() == 'RW':
        ## parameters[0] = alpha, relCue = 2, irrelCue = 3, relCueCol = 4, targetLoc = 5, data
        Q_est, selcues, pe = dms.frw_1c(parameters = parameters, data = data)
        
    elif model.upper() == 'HYBRID':
        ## parameters[0] = eta, relCue = 2, irrelCue = 3, relCueCol = 4, targetLoc = 5, data, salpha = 0.01
        Q_est, selcues, pe = dms.fhybrid_1c(parameters = parameters, data = data)

    ## Estimated value of selected cue
    EstSel = np.full((data.shape[0], 1), np.nan)
    ## Prediction error of selected cue
    peSel = np.full((data.shape[0], 1), np.nan)
    for trial in range(data.shape[0]):
        EstSel[trial] = Q_est[trial, selcues[trial]]
        peSel[trial] = abs(pe[trial])
    ## Reaction times
    RT = data[:, 12].reshape((-1, 1))

    # Correlation between RS and PE
    return - stats.spearmanr(RT, peSel, nan_policy = 'omit')[0]


def fModelSpearCorPos(parameters=(0.01, 0.5), model='RW', data=SimData):
    """
    Negative Spearman correlation of learning model
    NaN omited.
    Fast models.
    Signed prediction errors.
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
        Simulated data that has to be optimised
        The default is SimData.

    Returns
    -------
    negative spearman r

    """


    # Estimated values
    #-----------------
    if model.upper() == 'RW':
        ## parameters[0] = alpha, relCue = 2, irrelCue = 3, relCueCol = 4, targetLoc = 5, data
        Q_est, selcues, pe = dms.frw_1c(parameters = parameters, data = data)
        
    elif model.upper() == 'HYBRID':
        ## parameters[0] = eta, relCue = 2, irrelCue = 3, relCueCol = 4, targetLoc = 5, data, salpha = 0.01
        Q_est, selcues, pe = dms.fhybrid_1c(parameters = parameters, data = data)

    ## Estimated value of selected cue
    EstSel = np.full((data.shape[0], 1), np.nan)
    ## Prediction error of selected cue
    peSel = np.full((data.shape[0], 1), np.nan)
    for trial in range(data.shape[0]):
        EstSel[trial] = Q_est[trial, selcues[trial]]
        peSel[trial] = pe[trial]
    ## Reaction times
    RT = data[:, 12].reshape((-1, 1))

    # Correlation between RS and PE
    return stats.spearmanr(RT, peSel, nan_policy = 'omit')[0]



#%% ~~ Dataframe ~~ %%#


column_list = ['id', 'nsim', 'simAlpha', 'simBeta', 'guessAlpha', 'guessBeta',
               'cors_pre', 'p_pre', 'cors_abs_pre', 'p_abs_pre',
               'opt_alpha_na', 'opt_beta_na', 'opt_alpha_abs', 'opt_beta_abs', 'opt_alpha_pos', 'opt_beta_pos',
               'cors_post_na', 'p_post_na', 'cors_post_abs', 'p_post_abs', 'cors_post_pos', 'p_post_pos']

data = pd.DataFrame(columns=column_list)


column_list_pre = ['participant', 'guessAlpha', 'guessBeta', 'RT',
                   'Qest_pre', 'selcue_pre', 'pe_pre']
datapre = pd.DataFrame(columns=column_list_pre)

column_list_post = ['participant', 'guessAlpha', 'guessBeta', 'RT',
                    'Qest_post_na', 'selcue_post_na', 'pe_post_na',
                    'Qest_post_abs', 'selcue_post_abs', 'pe_post_abs']
datapost = pd.DataFrame(columns=column_list_post)



#%% ~~ Optimisation ~~ %%#
##########################


indexnr = 0
easypre = 0
easypost = 0

# Non-absolute values
## Before
cors_pre = np.zeros(2)
## After
cors_post_na = np.zeros(2)

# Absolute values
## Before
cors_abs_pre = np.zeros(2)
## After
cors_post_abs = np.zeros(2)

# Positive non-absolute values
## Before
cors_pre
## After
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
        for file in range(nsim):
            SimData = np.genfromtxt(simList[file], delimiter = ',')
            ## Remove titles (or find a way that it is not NaN)
            SimData = SimData[1:, 1:]
            file_info = simList[file].split('_')
            print(file,  betaOptions[beta], alphaOptions[alpha])


            # Pre optimization
            ##################
            if opt_model.upper() == 'RW':
                Qest_pre, selcue_pre, pe_pre = dms.frw_1c(parameters = x0, data = SimData)
            elif opt_model.upper() == 'HYBRID':
                Qest_pre, selcue_pre, pe_pre = dms.fhybrid_1c(parameters = x0, data = SimData)
            
            ## Store dataframe pre optimisation
            for triali in range(SimData.shape[0]):
                datapre.loc[easypre] = [file, alphaOptions[alpha], betaOptions[beta], SimData[triali, 20],
                                        Qest_pre[triali, selcue_pre[triali]], selcue_pre[triali], pe_pre[triali]]
                easypre += 1
            
            # if file < 3:
            #     # Plot
            #     # Plot number
            #     plt.figure(plotnr)
            #     plt.title(f'Cue selection {file_info[-2]}')
            #         ## Cue estimates
            #     plt.plot(Qest_pre[:, 0], label = 'Cue 1')
            #     plt.plot(Qest_pre[:, 1], label = 'Cue 2')
            #         ## Selected cue
            #     plt.plot(selcue_pre[:], label = 'Selected cue')
            #         ## True cue
            #     plt.plot(SimData[:, 4], label = 'True cue', linestyle = '-.')
            #         ## Legend
            #     plt.legend()
                
            #     plt.show()
            #     plotnr += 1
            
            # Correlation PE-RT
            #------------------
            ## Reaction times
            RT = SimData[:, 12].reshape((-1, 1))
            cors_pre = stats.spearmanr(RT, pe_pre, nan_policy = 'omit')
            cors_abs_pre = stats.spearmanr(RT, abs(pe_pre), nan_policy = 'omit')


            # Optimization
            ##############
            ## Non-absolute values
            estPar_na = optimize.fmin(fModelSpearCor, x0, args = (opt_model, SimData), ftol = 0.001)
            ## Absolute values
            estPar_abs = optimize.fmin(fModelSpearCorAbs, x0, args = (opt_model,SimData), ftol = 0.001)
            ## Positive non-absolute values
            estPar_pos = optimize.fmin(fModelSpearCorPos, x0, args = (opt_model, SimData), ftol = 0.001)


            # Post optimization
            ###################
            if opt_model.upper() == 'RW':
                # Non-absolute values
                Qest_post_na, selcue_post_na, pe_post_na = dms.frw_1c(parameters = estPar_na, data = SimData)
                # Absolute values
                Qest_post_abs, selcue_post_abs, pe_post_abs = dms.frw_1c(parameters = estPar_abs, data = SimData)
                # Positive non-absolute values
                Qest_post_pos, selcue_post_pos, pe_post_pos = dms.frw_1c(parameters = estPar_pos, data = SimData)
            elif opt_model.upper() == 'HYBRID':
                # Non-absolute values
                Qest_post_na, selcue_post_na, pe_post_na = dms.fhybrid_1c(parameters = estPar_na, data = SimData)
                # Absolute values
                Qest_post_abs, selcue_post_abs, pe_post_abs = dms.fhybrid_1c(parameters = estPar_abs, data = SimData)
                # Positive non-absolute values
                Qest_post_pos, selcue_post_pos, pe_post_pos = dms.fhybrid_1c(parameters = estPar_pos, data = SimData)
            
            ## Store dataframe post optimisation
            for triali in range(SimData.shape[0]):
                datapost.loc[easypost] = [file, alphaOptions[alpha], betaOptions[beta], SimData[triali, 20],
                                        Qest_post_na[triali, selcue_post_na[triali]], selcue_post_na[triali], pe_post_na[triali],
                                        Qest_post_abs[triali, selcue_post_abs[triali]], selcue_post_abs[triali], pe_post_abs[triali]]
                easypost += 1
            
            # if file < 3:
            #     # Selection plot
            #     plt.figure(plotnr)
            #     plt.subplot(2, 1, 1)
            #     plt.title(f'Cue selection {file_info[-1]} post')
            #     plt.xlabel('SPE')
            #         ## Cue estimates
            #     plt.plot(Qest_post_na[:, 0], label = 'Cue 1')
            #     plt.plot(Qest_post_na[:, 1], label = 'Cue 2')
            #         ## Selected cue
            #     plt.plot(selcue_post_na[:], label = 'Sel Cue')
            #         ## True cue
            #     plt.plot(SimData[:, 4], label = 'True cue', linestyle = '-.')
            #         ## Legend
            #     plt.legend()
                
            #     plt.subplot(2, 1, 2)
            #     plt.xlabel('UPE')
            #         ## Cue estimates
            #     plt.plot(Qest_post_abs[:, 0], label = 'Cue 1')
            #     plt.plot(Qest_post_abs[:, 1], label = 'Cue 2')
            #         ## Selected cue
            #     plt.plot(selcue_post_abs[:], label = 'Sel Cue')
            #         ## True cue
            #     plt.plot(SimData[:, 4], label = 'True cue', linestyle = '-.')
            #         ## Legend
            #     plt.legend()
                
            #     plt.show()
            #     plotnr += 1
                
            #     plt.figure(plotnr)
            #     plt.title(f'Cue selection {file_info[-1]} pos / post')
            #         ## Cue estimates
            #     plt.plot(Qest_post_pos[:, 0], label = 'Cue 1')
            #     plt.plot(Qest_post_pos[:, 1], label = 'Cue 2')
            #         ## Selected cue
            #     plt.plot(selcue_post_pos[:], label = 'Sel Cue')
            #         ## True cue
            #     plt.plot(SimData[:, 4], label = 'True cue', linestyle = '-.')
            #         ## Legend
            #     plt.legend()
                
            #     plt.show()
            #     plotnr += 1
            
            # Correlation PE-RT
            #------------------
            ## Reaction times
            RT = SimData[:, 12].reshape((-1, 1))
            # print(stats.spearmanr(RT, pe_post_na, nan_policy = 'omit'))
            cors_post_na = stats.spearmanr(RT, pe_post_na, nan_policy = 'omit')
            cors_post_abs = stats.spearmanr(RT, abs(pe_post_abs), nan_policy = 'omit')
            cors_post_pos = stats.spearmanr(RT, pe_post_pos, nan_policy = 'omit')
            
            # if file < 3:
            # # Plot
            #     plt.figure(plotnr)
            #     plt.subplot(2, 1, 1)
            #     plt.title(f'Correlation: 1 cue / Pre-Post / {file_info[-1]} / pe')
            #     plt.xlabel('SPE')
            #     plt.scatter(RT, pe_pre, alpha = 0.5, label = 'Pre')
            #     plt.scatter(RT, pe_post_na, alpha = 0.5, label = 'Post')
            #     plt.legend()
                
            #     plt.subplot(2, 1, 2)
            #     plt.xlabel('UPE')
            #     plt.scatter(RT, pe_pre, alpha = 0.5, label = 'Pre')
            #     plt.scatter(RT, abs(pe_post_abs), alpha = 0.5, label = 'Post')
            #     plt.legend()
                
            #     plt.show()
            #     plotnr += 1
                
            #     plt.figure(plotnr)
            #     plt.title(f'Correlation: 1 cue / Pre-Post / {file_info[-1]} / pe / pos')
            #     plt.scatter(RT, pe_pre, alpha = 0.5, label = 'Pre')
            #     plt.scatter(RT, pe_post_pos, alpha = 0.5, label = 'Post')
                
            #     plt.show()
            #     plotnr += 1


            ## Store dataframe
            data.loc[indexnr] = [file, file_info[-2], file_info[-5], file_info[-3], alphaOptions[alpha], betaOptions[beta],
                                 cors_pre[0], cors_pre[1], cors_abs_pre[0], cors_abs_pre[1],
                                 estPar_na[0], estPar_na[1], estPar_abs[0], estPar_abs[1], estPar_pos[0], estPar_pos[1],
                                 cors_post_na[0], cors_post_na[1], cors_post_abs[0], cors_post_abs[1], cors_post_pos[0], cors_post_pos[1]]
            indexnr += 1


        end_beta = time.time()
        print(f'Duration beta {beta}: {round((end_beta - start_beta) / 60, 2)} minutes')
    end_alpha = time.time()
    print(f'Duration alpha {alpha}: {round((end_alpha - start_alpha) / 60, 2)} minutes')
end_total = time.time()
print(f'Total duration: {round((end_total - start_total) / 60, 2)} minutes')


data.to_csv(f'PEsim_{opt_model}.csv', index = False)
datapre.to_csv(f'PEsim_{opt_model}_pre.csv', index = False)
datapost.to_csv(f'PEsim_{opt_model}_post.csv', index = False)



#------------------------------------------------------------------------- End
