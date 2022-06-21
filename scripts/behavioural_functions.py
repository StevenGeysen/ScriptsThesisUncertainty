#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Model functions -- Version 1
Last edit:  2022/06/21
Author(s):  Geysen, Steven (SG)
Notes:      - Models for the ananlysis of behavioural data from
                Marzecova et al. (2019)
            - Models with SoftMax policy
                * Rescorla-Wagner (Daphne)
                * Rescorla-Wagner - Pearce-Hall hybrid (Hugo)
            - Release notes:
                * Copied models from sim_functions
To do:      - Adjust models to behavioural data
            - Argmax
Questions:  - How do I update the models? Do I use the validity of the
                participant's selection, or do I use the validity of the
                model's selection?
            - How do I know the participant's selection?
Comments:   SG: Models return pandas.DataFrame. Model functions return
                originale data with model selection appended.
            
Sources:    https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00612/full
            https://docs.sympy.org/latest/modules/stats.html
            https://github.com/Kingsford-Group/ribodisomepipeline/blob/master/scripts/exGaussian.py
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html
            https://hannekedenouden.ruhosting.nl/RLtutorial/html/SimulationTopPage.html
            https://lindeloev.shinyapps.io/shiny-rt/
            https://elifesciences.org/articles/49547
"""



#%% ~~ Imports ~~ %%#



import numpy as np
import pandas as pd

from scipy import stats



#%% ~~ Models ~~ %%#
####################


#%% ~~ Models (1 cue) ~~ %%#
#--------------------------#


# ~~ Rescorla - Wagner ~~ #
def ppRW_1c(parameters, data):
    """
    Rescorla-Wagner predictions based on participants behaviour
    The Rescorla - Wagner learning model with SoftMax. The policy calculations
    are part of the function (and not a separate function). Only the cue
    estimate of the selected cue is updated.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is the learning rate in the model.
        Second parameter is constant in action selection method.
        The default is (0.01, 0.5).
    data : pandas.DataFrame
        Dataframe containing the structure of the experiment.

    Returns
    -------
    ppData : pandas.DataFrame
        Contains columns of 'data' and simulated behaviour of Daphne:
            0. 'selCue_RW' - selected cue by the Rescorla-Wagner model
            1. 'rt_RW' - response time
            2. 'reward_RW' - reward based on cue selected by Rescorla-Wagner
            3. 'RPE_RW' - reward prediction error
            4. 'Qest_0_RW' - estimated value of cue 0
            5. 'Qest_1_RW' - estimated value of cue 1

    """


    # Variables
    #----------
    # Dataframe
    var_list = [
        'selCue_RW', 'rt_RW', 'reward_RW',
        'RPE_RW', 'Qest_0_RW', 'Qest_1_RW'
        ]
    ppDict = {vari:[] for vari in var_list}
    
    ## Number of trials
    n_trials = data.shape[0]
    ## Number of cues
    N_CUES = 2
    # Model
    ## Estimated value of cue
    Q_est = np.full((n_trials, N_CUES), 1/N_CUES)
    
    # Policy
    ## Selected cue
    selcue = np.nan

    # Trial loop
    for triali, trial in data.iterrows():
        # Policy
        #-------
        ## Random for first trial
        if triali == 0:
            selcue = np.random.randint(N_CUES)
            probcue = 0.5
        else:
            ## Probability of cue 0
            probcue = np.exp(parameters[1] * Q_est[triali, 0]) / \
                np.sum(np.exp(np.multiply(parameters[1], Q_est[triali, :])))
            ##SG: If the probability of cue 0 is smaller than a random value,
                # follow cue 1.
            temp = np.random.rand() <= probcue
            ## Action selection
            selcue = int(temp == 0)
        ppDict['selCue_RW'].append(selcue)
        
        # Response time
        try:
            ##SG: K = tau / sigma, loc = mu, scale = sigma. Sigma and mu are
                # taken from exgauss fit of the original data.
            RT = stats.exponnorm.rvs(K = abs(selcue - probcue) / 0.02635,
                                     loc = 0.3009, scale = 0.02635)
        except:
            RT = np.nan
            print('Failed rt sampling')
        if RT == float('inf'):
            RT = 1.7
        ppDict['rt_RW'].append(RT)
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcue == trial.targetLoc)
        ppDict['reward_RW'].append(reward)
        
        # Update rule (RW)
        #-----------------
        if triali == 0:
            # Reward prediction error
            rpe = reward - Q_est[triali, selcue]
            # Cue estimates
            Q_est[triali, selcue] = Q_est[triali, selcue] + parameters[0] * rpe
        else:
            # Reward prediction error
            rpe = reward - Q_est[triali - 1, selcue]
            # Cue estimates
            ## Repeat cue estimates of previous trial
            Q_est[triali, :] = Q_est[triali - 1, :]
            ## Update cue estimate of selected stimulus in current trial
            Q_est[triali, selcue] = Q_est[triali, selcue] + parameters[0] * rpe
        ppDict['RPE_RW'].append(rpe)
        for qi, q_est in enumerate(Q_est[triali, :]):
            ppDict[f'Qest_{qi}_RW'].append(q_est)
    
    # Save data
    ppData = pd.DataFrame(ppDict, columns=var_list)
    ## Correct indexes
    ppData.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    ppData = pd.concat([data, ppData], axis=1)

    return ppData


# ~~ RW-PH Hybrid ~~ #
def ppHybrid_1c(parameters, data, salpha=0.01):
    """
    Hugo the hybrid learner
    The RW-PH hybrid learning model with SoftMax. The policy calculations
    are part of the function (and not a separate function). Only the cue
    estimate of the selected cue is updated.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is eta.
        Second parameter is constant in action selection method.
        The default is (0.01, 0.5).
    data : pandas.DataFrame
        Dataframe containing the structure of the experiment.
    salpha : TYPE, optional
        Start of alpha. The default is 0.01.

    Returns
    -------
    ppData : pandas.DataFrame
        Contains columns of 'data' and simulated behaviour of Hugo:
            0. 'selCue_hyb' - selected cue by the hybrid model
            1. 'rt_hyb' - response time
            2. 'reward_hyb' - reward based on cue selected by hybrid model
            3. 'alpha_hyb' - learning rate
            4. 'RPE_hyb' - reward prediction error
            5. 'Qest_0_hyb' - estimated value of cue 0
            6. 'Qest_1_hyb' - estimated value of cue 1

    """


    # Variables
    #----------
    # Dataframe
    var_list = [
        'selCue_hyb', 'rt_hyb', 'reward_hyb', 'alpha_hyb',
        'RPE_hyb', 'Qest_0_hyb', 'Qest_1_hyb'
        ]
    ppDict = {vari:[] for vari in var_list}
    
    ## Number of trials
    n_trials = data.shape[0]
    ## Number of cues
    N_CUES = 2
    
    # Model
    ## Alpha
    alpha = np.full((n_trials, N_CUES), salpha)
    ## Estimated value of cue
    Q_est = np.full((n_trials, N_CUES), 1/N_CUES)
    
    # Policy
    ## Selected cue
    selcue = np.nan

    # Trial loop
    for triali, trial in data.iterrows():
        # Select cue
        ## Random for first trial
        if triali == 0:
            selcue = np.random.randint(N_CUES)
            probcue = 0.5
        else:
            ## Probability of cue 0
            probcue = np.exp(parameters[1] * Q_est[triali, 0]) / \
                np.sum(np.exp(np.multiply(parameters[1], Q_est[triali, :])))
            ##SG: If the probability of cue 0 is smaller than a random value,
                # follow cue 1.
            temp = np.random.rand() <= probcue
            ## Action selection
            selcue = int(temp == 0)
        ppDict['selCue_hyb'].append(selcue)
        
        # Response time
        try:
            ##SG: K = tau / sigma, loc = mu, scale = sigma. Sigma and mu are
                # taken from exgauss fit of the original data.
            RT = stats.exponnorm.rvs(K = abs(selcue - probcue) / 0.02635,
                                     loc = 0.3009, scale = 0.02635)
        except:
            RT = np.nan
            print('Failed rt sampling')
        if RT == float('inf'):
            RT = 1.7
        ppDict['rt_hyb'].append(RT)
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcue == trial.targetLoc)
        ppDict['reward_hyb'].append(reward)
        
        # Hybrid
        #-------
        if triali == 0:
            # Reward prediction error
            rpe = reward - Q_est[triali, selcue]
            # Alpha (PH)
            alpha[triali, selcue] = parameters[0] * np.abs(rpe) + \
                (1 - parameters[0]) * alpha[triali, selcue]
            # Cue estimates
            Q_est[triali, selcue] = Q_est[triali, selcue] + \
                alpha[triali, selcue] * rpe
        else:
            # Reward prediction error
            rpe = reward - Q_est[triali - 1, selcue]
            # Update values of selected stimulus in current trial
            alpha[triali, :] = alpha[triali - 1, :]
            Q_est[triali, :] = Q_est[triali - 1, :]
            # Alpha (PH)
            alpha[triali, selcue] = parameters[0] * np.abs(rpe) + \
                (1 - parameters[0]) * alpha[triali, selcue]
            # Cue estimates
            Q_est[triali, selcue] = Q_est[triali, selcue] + \
                alpha[triali, selcue] * rpe
        ppDict['RPE_hyb'].append(rpe)
        ppDict['alpha_hyb'].append(alpha[triali, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            ppDict[f'Qest_{qi}_hyb'].append(q_est)
    
    # Save data
    ppData = pd.DataFrame(ppDict, columns=var_list)
    ## Correct indexes
    ppData.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    ppData = pd.concat([data, ppData], axis=1)

    return ppData



#%% ~~ Models (2 cues) ~~ %%#
#---------------------------#


# ~~ Rescorla - Wagner ~~ #
def ppRW_2c(parameters, data):
    """
    Daphne the delta learner
    The Rescorla - Wagner learning model with SoftMax. The policy calculations
    are part of the function (and not a separate function). The cue
    estimates of both cues are updated.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is the learning rate in the model.
        Second parameter is constant in action selection method.
        The default is (0.01, 0.5).
    data : pandas.DataFrame
        Dataframe containing the structure of the experiment.

    Returns
    -------
    simData : pandas.DataFrame
        Contains columns of 'data' and simulated behaviour of Daphne:
            0. 'selCue_RW' - selected cue by the Rescorla-Wagner model
            1. 'rt_RW' - response time
            2. 'reward_RW' - reward based on cue selected by Rescorla-Wagner
            3. 'RPE_RW' - reward prediction error
            4. 'Qest_0_RW' - estimated value of cue 0
            5. 'Qest_1_RW' - estimated value of cue 1

    """


    # Variables
    #----------
    # Dataframe
    var_list = [
        'selCue_RW', 'rt_RW', 'reward_RW',
        'RPE_RW', 'Qest_0_RW', 'Qest_1_RW'
        ]
    ppDict = {vari:[] for vari in var_list}
    
    ## Number of trials
    n_trials = data.shape[0]
    ## Number of cues
    N_CUES = 2
    # Model
    ## Estimated value of cue
    Q_est = np.full((n_trials, N_CUES), 1/N_CUES)
    
    # Policy
    ## Selected cue
    selcue = np.nan

    # Trial loop
    for triali, trial in data.iterrows():
        # Policy
        #-------
        ## Random for first trial
        if triali == 0:
            selcue = np.random.randint(N_CUES)
            probcue = 0.5
        else:
            ## Probability of cue 0
            probcue = np.exp(parameters[1] * Q_est[triali, 0]) / \
                np.sum(np.exp(np.multiply(parameters[1], Q_est[triali, :])))
            ##SG: If the probability of cue 0 is smaller than a random value,
                # follow cue 1.
            temp = np.random.rand() <= probcue
            ## Action selection
            selcue = int(temp == 0)
        ppDict['selCue_RW'].append(selcue)
        
        # Response time
        try:
            ##SG: K = tau / sigma, loc = mu, scale = sigma. Sigma and mu are
                # taken from exgauss fit of the original data.
            RT = stats.exponnorm.rvs(K = abs(selcue - probcue) / 0.02635,
                                     loc = 0.3009, scale = 0.02635)
        except:
            RT = np.nan
            print('Failed rt sampling')
        if RT == float('inf'):
            RT = 1.7
        ppDict['rt_RW'].append(RT)
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcue == trial.targetLoc)
        ppDict['reward_RW'].append(reward)
        
        # Update rule (RW)
        #-----------------
        if triali == 0:
            for cuei in range(N_CUES):
                # Reward prediction error
                rpe = reward - Q_est[triali, cuei]
                # Cue estimates
                Q_est[triali, cuei] = Q_est[triali, cuei] + parameters[0] * rpe
        else:
            # Repeat cue estimates of previous trial
            Q_est[triali, :] = Q_est[triali - 1, :]
            for cuei in range(N_CUES):
                # Reward prediction error
                rpe = reward - Q_est[triali - 1, cuei]
                # Cue estimates
                Q_est[triali, cuei] = Q_est[triali, cuei] + parameters[0] * rpe
        ppDict['RPE_RW'].append(reward - Q_est[triali - 1, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            ppDict[f'Qest_{qi}_RW'].append(q_est)
    
    # Save data
    ppData = pd.DataFrame(ppDict, columns=var_list)
    ## Correct indexes
    ppData.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    ppData = pd.concat([data, ppData], axis=1)

    return ppData


# ~~ RW-PH Hybrid ~~ #
def ppHybrid_2c(parameters, data, salpha=0.01):
    """
    Hugo the hybrid learner
    The RW-PH hybrid learning model with SoftMax. The policy calculations
    are part of the function (and not a separate function). The cue
    estimates of both cues are updated.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is eta.
        Second parameter is constant in action selection method.
        The default is (0.01, 0.5).
    data : pandas.DataFrame
        Dataframe containing the structure of the experiment.
    salpha : TYPE, optional
        Start of alpha. The default is 0.01.

    Returns
    -------
    simData : pandas.DataFrame
        Contains columns of 'data' and simulated behaviour of Hugo:
            0. 'selCue_hyb' - selected cue by the hybrid model
            1. 'rt_hyb' - response time
            2. 'reward_hyb' - reward based on cue selected by hybrid model
            3. 'alpha_hyb' - learning rate
            4. 'RPE_hyb' - reward prediction error
            5. 'Qest_0_hyb' - estimated value of cue 0
            6. 'Qest_1_hyb' - estimated value of cue 1

    """


    # Variables
    #----------
    # Dataframe
    var_list = [
        'selCue_hyb', 'rt_hyb', 'reward_hyb', 'alpha_hyb',
        'RPE_hyb', 'Qest_0_hyb', 'Qest_1_hyb'
        ]
    ppDict = {vari:[] for vari in var_list}
    
    ## Number of trials
    n_trials = data.shape[0]
    ## Number of cues
    N_CUES = 2
    
    # Model
    ## Alpha
    alpha = np.full((n_trials, N_CUES), salpha)
    ## Estimated value of cue
    Q_est = np.full((n_trials, N_CUES), 1/N_CUES)
    
    # Policy
    ## Selected cue
    selcue = np.nan

    # Trial loop
    for triali, trial in data.iterrows():
        # Select cue
        ## Random for first trial
        if triali == 0:
            selcue = np.random.randint(N_CUES)
            probcue = 0.5
        else:
            ## Probability of cue 0
            probcue = np.exp(parameters[1] * Q_est[triali, 0]) / \
                np.sum(np.exp(np.multiply(parameters[1], Q_est[triali, :])))
            ##SG: If the probability of cue 0 is smaller than a random value,
                # follow cue 1.
            temp = np.random.rand() <= probcue
            ## Action selection
            selcue = int(temp == 0)
        ppDict['selCue_hyb'].append(selcue)
        
        # Response time
        try:
            ##SG: K = tau / sigma, loc = mu, scale = sigma. Sigma and mu are
                # taken from exgauss fit of the original data.
            RT = stats.exponnorm.rvs(K = abs(selcue - probcue) / 0.02635,
                                     loc = 0.3009, scale = 0.02635)
        except:
            RT = np.nan
            print('Failed rt sampling')
        if RT == float('inf'):
            RT = 1.7
        ppDict['rt_hyb'].append(RT)
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcue == trial.targetLoc)
        ppDict['reward_hyb'].append(reward)
        
        # Hybrid
        #-------
        if triali == 0:
            for cuei in range(N_CUES):
                # Reward prediction error
                rpe = reward - Q_est[triali, cuei]
                # Alpha (PH)
                alpha[triali, cuei] = parameters[0] * np.abs(rpe) + \
                    (1 - parameters[0]) * alpha[triali, cuei]
                # Cue estimates
                Q_est[triali, cuei] = Q_est[triali, cuei] + \
                    alpha[triali, cuei] * rpe
        else:
            ## Repeat values of previous trial
            Q_est[triali, :] = Q_est[triali - 1, :]
            for cuei in range(N_CUES):
                # Reward prediction error
                rpe = reward - Q_est[triali - 1, cuei]
                # Alpha (PH)
                alpha[triali, cuei] = parameters[0] * np.abs(rpe) + \
                    (1 - parameters[0]) * alpha[triali, cuei]
                # Cue estimates
                Q_est[triali, cuei] = Q_est[triali, cuei] + \
                    alpha[triali, cuei] * rpe
        ppDict['RPE_hyb'].append(reward - Q_est[triali - 1, selcue])
        ppDict['alpha_hyb'].append(alpha[triali, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            ppDict[f'Qest_{qi}_hyb'].append(q_est)
    
    # Save data
    ppData = pd.DataFrame(ppDict, columns=var_list)
    ## Correct indexes
    ppData.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    ppData = pd.concat([data, ppData], axis=1)

    return ppData



#%% ~~ Controle models ~~ %%#
#---------------------------#


# ~~ Wilhelm ~~ #
def ppWSLS(data):
    """
    Wilhelm the Win-stay-lose-shift model
    Selects the cue that was predictive during its previous presentation.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the structure of the experiment.

    Returns
    -------
    simData : pd.DataFrame
        Contains columns of 'data' and simulated behaviour of Wilhelm:
            0. 'selCue_W' - selected cue by the WSLS model
            1. 'reward_W' - reward based on cue selected by WSLS model

    """


    # Variables
    # ---------
    # Dataframe
    var_list = ['selCue_W', 'reward_W']
    ppDict = {vari:[] for vari in var_list}
    
    # Model
    ## Number of trials
    n_trials = data.shape[0]
    ## Number of cues
    N_CUES = 2
    ## Reward
    reward = np.nan
    ## Selected cues
    selcues = np.full((n_trials, ), np.nan)

    # Trial loop
    for triali, trial in data.iterrows():
        # Policy
        #-------
        ## Random for first trial
        if triali == 0:
            selcues[triali] = np.random.randint(N_CUES)
        else:
            if reward == 1:
                selcues[triali] = selcues[triali - 1]
            else:
                selcues[triali] = 1 - selcues[triali - 1]
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcues[triali] == trial.targetLoc)
        
        ppDict['selCue_W'].append(selcues[triali])
        ppDict['reward_W'].append(reward)
    
    # Save data
    ppData = pd.DataFrame(ppDict, columns=var_list)
    ## Correct indexes
    ppData.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    ppData = pd.concat([data, ppData], axis=1)

    # Output dataframe
    return ppData


# ~~ Renee ~~ #
def ppRandom(data):
    """
    Renee the random model
    Random cue selection.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the structure of the experiment.

    Returns
    -------
    simData : pd.DataFrame
        Contains columns of 'data' and simulated behaviour of Renee:
            0. 'selCue_R' - selected cue by the random model
            1. 'reward_R' - reward based on cue selected by random model

    """


    # Variables
    # ---------
    # Dataframe
    var_list = ['selCue_R', 'reward_R']
    ppDict = {vari:[] for vari in var_list}
    
    # Number of cues
    N_CUES = 2

    # Trial loop
    for triali, trial in data.iterrows():
        selcue = np.random.randint(N_CUES)
        ppDict['selCue_R'].append(selcue)
        reward = int(selcue == trial.targetLoc)
        ppDict['reward_R'].append(reward)
    
    # Save data
    ppData = pd.DataFrame(ppDict, columns=var_list)
    ## Correct indexes
    ppData.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    ppData = pd.concat([data, ppData], axis=1)

    # Output dataframe
    return ppData



#%% ~~ Others ~~ %%#
####################



#------------------------------------------------------------------------- End
