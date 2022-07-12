#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Model functions -- Version 3
Last edit:  2022/07/12
Author(s):  Geysen, Steven (SG)
Notes:      - Models for the ananlysis of behavioural data from
                Marzecova et al. (2019)
                * Models
                    - Rescorla-Wagner (Daphne)
                    - Rescorla-Wagner - Pearce-Hall hybrid (Hugo)
                    - Win-stay-lose-shift (Wilhelm)
                    - Random (Renee)
                * Negative Spearman correlation
            - Release notes:
                * Removed (negaitve) log likelihood
To do:      - Adjust models to behavioural data
            
Questions:  
Comments:   SG: Models return pandas.DataFrame. Model functions return
                originale data with model selection appended.
            
Sources:    https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00612/full
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html
            https://lindeloev.shinyapps.io/shiny-rt/
            https://elifesciences.org/articles/49547
"""



#%% ~~ Imports ~~ %%#



import numpy as np
import pandas as pd

import fns.assisting_functions as af

from scipy import stats



#%% ~~ Blocks ~~ %%#
####################


def pp_models():
    """
    Participant dictionary
    Contains the functions of the different behavioural models.
    """

    return {'RW': ppRW_1c,
            'RW2': ppRW_2c,
            'H': ppHybrid_1c,
            'H2': ppHybrid_2c,
            'W': ppWSLS,
            'R': ppRandom}



#%% ~~ Models ~~ %%#
####################


#%% ~~ Models (1 cue) ~~ %%#
#--------------------------#


# ~~ Rescorla - Wagner ~~ #
def ppRW_1c(parameters, data, asm='soft'):
    """
    Daphne the delta learner
    Rescorla-Wagner predictions learning model with SoftMax, based on
    participants' behaviour. The policy calculations are part of the function
    (and not a separate function). Only the cue estimate of the selected cue is
    updated.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is the learning rate in the model.
        Second parameter is constant in action selection method.
        The default is (0.01, 0.5).
    data : pandas.DataFrame
        Dataframe containing the structure of the experiment.
    asm : string, optional
        The action selection method, policy. The default is SoftMax.

    Returns
    -------
    ppData : pandas.DataFrame
        Contains columns of 'data' and simulated behaviour of Daphne:
            0. 'selCue_RW' - selected cue by the Rescorla-Wagner model
            1. 'prob_RW' - probability to select cue 0
            2. 'reward_RW' - reward based on cue selected by Rescorla-Wagner
            3. 'RPE_RW' - reward prediction error
            4. 'Qest_0_RW' - estimated value of cue 0
            5. 'Qest_1_RW' - estimated value of cue 1
    """

    # Variables
    # ---------
    # Dataframe
    var_list = [
        'selCue_RW', 'prob_RW', 'reward_RW',
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
        # ------
        ## Random for first trial
        if triali == 0:
            selcue = np.random.randint(N_CUES)
            probcue = 0.5
        else:
            selcue, probcue = af.policy(asm, Q_est[triali - 1, :],
                                        parameters[1])
        ppDict['selCue_RW'].append(selcue)
        ppDict['prob_RW'].append(probcue)
        
        # Reward
        # ------
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcue == trial.targetLoc)
        ppDict['reward_RW'].append(reward)
        
        # Update rule (RW)
        # ----------------
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

    return af.save_data(ppDict, data, var_list)


# ~~ RW-PH Hybrid ~~ #
def ppHybrid_1c(parameters, data, salpha=0.01, asm='soft'):
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
    salpha : float, optional
        Start of alpha. The default is 0.01.
    asm : string, optional
        The action selection method, policy. The default is SoftMax.

    Returns
    -------
    ppData : pandas.DataFrame
        Contains columns of 'data' and simulated behaviour of Hugo:
            0. 'selCue_H' - selected cue by the hybrid model
            1. 'prob_H' - probability to select cue 0
            2. 'reward_H' - reward based on cue selected by hybrid model
            3. 'alpha_H' - learning rate
            4. 'RPE_H' - reward prediction error
            5. 'Qest_0_H' - estimated value of cue 0
            6. 'Qest_1_H' - estimated value of cue 1
    """

    # Variables
    # ---------
    # Dataframe
    var_list = [
        'selCue_H', 'prob_H', 'reward_H', 'alpha_H',
        'RPE_H', 'Qest_0_H', 'Qest_1_H'
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
        # Policy
        # ------
        ## Random for first trial
        if triali == 0:
            selcue = np.random.randint(N_CUES)
            probcue = 0.5
        else:
            selcue, probcue = af.policy(asm, Q_est[triali - 1, :],
                                        parameters[1])
        ppDict['selCue_H'].append(selcue)
        ppDict['prob_H'].append(probcue)
        
        # Reward
        # ------
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcue == trial.targetLoc)
        ppDict['reward_H'].append(reward)
        
        # Hybrid
        # ------
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
        ppDict['RPE_H'].append(rpe)
        ppDict['alpha_H'].append(alpha[triali, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            ppDict[f'Qest_{qi}_H'].append(q_est)

    return af.save_data(ppDict, data, var_list)


#%% ~~ Models (2 cues) ~~ %%#
#---------------------------#


# ~~ Rescorla - Wagner ~~ #
def ppRW_2c(parameters, data, asm='soft'):
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
    asm : string, optional
        The action selection method, policy. The default is SoftMax.

    Returns
    -------
    simData : pandas.DataFrame
        Contains columns of 'data' and simulated behaviour of Daphne:
            0. 'selCue_RW2' - selected cue by the Rescorla-Wagner model
            1. 'prob_RW2' - probability to select cue 0
            2. 'reward_RW2' - reward based on cue selected by Rescorla-Wagner
            3. 'RPE_RW2' - reward prediction error
            4. 'Qest_0_RW2' - estimated value of cue 0
            5. 'Qest_1_RW2' - estimated value of cue 1
    """

    # Variables
    # ---------
    # Dataframe
    var_list = [
        'selCue_RW2', 'prob_RW2', 'reward_RW2',
        'RPE_RW2', 'Qest_0_RW2', 'Qest_1_RW2'
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
        # ------
        ## Random for first trial
        if triali == 0:
            selcue = np.random.randint(N_CUES)
            probcue = 0.5
        else:
            selcue, probcue = af.policy(asm, Q_est[triali - 1, :],
                                        parameters[1])
        ppDict['selCue_RW2'].append(selcue)
        ppDict['prob_RW2'].append(probcue)
        
        # Reward
        # ------
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcue == trial.targetLoc)
        ppDict['reward_RW2'].append(reward)
        
        # Update rule (RW)
        # ----------------
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
        ppDict['RPE_RW2'].append(reward - Q_est[triali - 1, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            ppDict[f'Qest_{qi}_RW2'].append(q_est)

    return af.save_data(ppDict, data, var_list)


# ~~ RW-PH Hybrid ~~ #
def ppHybrid_2c(parameters, data, salpha=0.01, asm='soft'):
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
    salpha : float, optional
        Start of alpha. The default is 0.01.
    asm : string, optional
        The action selection method, policy. The default is SoftMax.

    Returns
    -------
    simData : pandas.DataFrame
        Contains columns of 'data' and simulated behaviour of Hugo:
            0. 'selCue_H2' - selected cue by the hybrid model
            1. 'prob_H2' - probability to select cue 0
            2. 'reward_H2' - reward based on cue selected by hybrid model
            3. 'alpha_H2' - learning rate
            4. 'RPE_H2' - reward prediction error
            5. 'Qest_0_H2' - estimated value of cue 0
            6. 'Qest_1_H2' - estimated value of cue 1
    """

    # Variables
    # ---------
    # Dataframe
    var_list = [
        'selCue_H2', 'prob_H2', 'reward_H2', 'alpha_H2',
        'RPE_H2', 'Qest_0_H2', 'Qest_1_H2'
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
        # Policy
        # ------
        ## Random for first trial
        if triali == 0:
            selcue = np.random.randint(N_CUES)
            probcue = 0.5
        else:
            selcue, probcue = af.policy(asm, Q_est[triali - 1, :],
                                        parameters[1])
        ppDict['selCue_H2'].append(selcue)
        ppDict['prob_H2'].append(probcue)
        
        # Reward
        # ------
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcue == trial.targetLoc)
        ppDict['reward_H2'].append(reward)
        
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
        ppDict['RPE_H2'].append(reward - Q_est[triali - 1, selcue])
        ppDict['alpha_H2'].append(alpha[triali, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            ppDict[f'Qest_{qi}_H2'].append(q_est)

    return af.save_data(ppDict, data, var_list)



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
            1. 'prob_W' - probability to select cue 0
            2. 'reward_W' - reward based on cue selected by WSLS model
    """

    # Variables
    # ---------
    # Dataframe
    var_list = ['selCue_W', 'prob_W', 'reward_W']
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
        # ------
        ## Random for first trial
        if triali == 0:
            selcues[triali] = np.random.randint(N_CUES)
        else:
            if reward == 1:
                selcues[triali] = selcues[triali - 1]
            else:
                selcues[triali] = 1 - selcues[triali - 1]
        # Reward
        # ------
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcues[triali] == trial.targetLoc)
        
        ppDict['selCue_W'].append(selcues[triali])
        ##SG: If there is some uncertainty in which cue Wilhelm will pick,
            # there is something wrong in the model.
        ppDict['prob_W'].append(1)
        ppDict['reward_W'].append(reward)

    return af.save_data(ppDict, data, var_list)


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
            1. 'prob_R' - probability to select cue 0
            2. 'reward_R' - reward based on cue selected by random model
    """

    # Variables
    # ---------
    # Dataframe
    var_list = ['selCue_R', 'prob_R', 'reward_R']
    ppDict = {vari:[] for vari in var_list}
    
    # Number of cues
    N_CUES = 2

    # Trial loop
    for triali, trial in data.iterrows():
        selcue = np.random.randint(N_CUES)
        ppDict['selCue_R'].append(selcue)
        ppDict['prob_R'].append(0.5)
        reward = int(selcue == trial.targetLoc)
        ppDict['reward_R'].append(reward)

    return af.save_data(ppDict, data, var_list)



#%% ~~ Others ~~ %%#
####################


def pp_negSpearCor(thetas, data, model):
    """
    Negative Spearman correlation of learning model of reaction time and
    estimated value of selected cue

    Parameters
    ----------
    thetas : list, array, tuple
        Parameter values.
    data : pd.DataFrame
        Prepared data of simulation, containing structure of experiment.
        Doesn't need to go through rescon first.
    model : string
        Name of the used model:
            RW - Rescorla-Wagner
            H - RW-PH hybrid
            W - Win-stay-lose-shift
            R - Random

    Returns
    -------
    negative spearman r
    """

    model = model.upper()
    # Wilhelm and Renee do not have reward prediction errors.
    assert not model in ['W', 'R'], 'Model has no RPE'

    # Rename to avoid duplicates
    data = data.rename(columns={f'selCue_{model}': 'selCue',
                                f'prob_{model}': 'prob',
                                f'RPE_{model}': 'RPE'})
    # Simulate data
    modelDict = pp_models()
    simData = modelDict[model](thetas, data)

    # Correlation between RT and RPE
    return - stats.spearmanr(simData['RT'].to_numpy(),
                             simData[f'RPE_{model}'].to_numpy(),
                             nan_policy = 'omit')[0]



#------------------------------------------------------------------------- End
