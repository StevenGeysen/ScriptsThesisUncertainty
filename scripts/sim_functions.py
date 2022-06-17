#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Simulation functions -- Version 1
Last edit:  2022/06/17
Author(s):  Geysen, Steven (SG)
Notes:      - Functions used for the simulation of the task used by
                Marzecova et al. (2019). Both structure as models.
            - Models with SoftMax policy
                * Rescorla-Wagner
                * Hybrid (Rescorla-Wagner - Pearce-Hall)
            - Release notes:
                * Added model functions to task structure
                
            
To do:      - Implement in other scripts
            - Argmax
            
Comments:   SG: Simulations return pandas.DataFrame. Model functions return
                originale data with model selection appended.
            
Sources:    
"""



#%% ~~ Imports ~~ %%#



import numpy as np
import pandas as pd



#%% ~~ Function ~~ %%#
######################


def sim_experiment(ppnr=1, ntrials=640, nswitch=7):
    """
    Simulate the eperimental structure of Marzecova et al. (2019)

    Parameters
    ----------
    ppnr : int, optional
        Participant number. Used for the randomisation of the reward
            probabilities.
        The default is 1.
    ntrials : int, optional
        Number of trials.
        The default is 640.
    nswitch : int, optional
        Number of switches in cue validity during the experiment.
        The default is 7.

    Returns
    -------
    data : pandas.DataFrame
        Contains:
            0. 'id' - participants id
            1. 'trial' - trial # of the task
            2. 'relCueCol' - colour of the relevant cue (0: white / 1: black)
            3. 'relCue' - direction of the relevant cue (0: left / 1: right)
            4. 'irrelCue'- direction of the irrelevant cue (0: left / 1: right)
            5. 'targetLoc' - location of the target (0: left / 1: right)
            6. 'validity' - trial validity

    """

    # Variables
    #----------
    # DataFrame
    column_list = [
        'id', 'trial',
        'relCueCol', 'relCue', 'irrelCue', 'targetLoc', 'validity'
        ]
    dataDict = {keyi: [] for keyi in column_list}
    dataDict['id'] = ppnr
    
    # Task
    n_stim = 2
    ## Colour of relevant cue
    relCueCol = np.random.randint(n_stim)
    ## Probabilities of reward for each stim
    prob = np.full(n_stim, 0.5)
    ## SG: For participants with an even number is the initial probability
        ## of the relevant cue 0.7
    if ppnr % 2 == 0:
        prob[relCueCol] = 0.7
    else:
        prob[relCueCol] = 0.85
    ## Trials where probability is switched
    switch = np.cumsum(np.random.randint(40, 120, size = nswitch))


    # Trial loop
    #-----------
    for triali in range(ntrials):
        dataDict['trial'] = triali
        # Switch probabilities
        if switch.shape[0] != 0:
            if triali == switch[0]:
                relCueCol = 1 - relCueCol
                
                if ppnr % 2 == 0:
                    ##SG: For participants with an even number is in the
                        # second half of trials the probability
                        # of the relevant cue 0.85.
                    if triali >= ntrials / 2:
                        prob[relCueCol] = 0.85
                    else:
                        prob[relCueCol] = 0.7
                else:
                    if triali >= ntrials / 2:
                        prob[relCueCol] = 0.7
                    else:
                        prob[relCueCol] = 0.85
                prob[1 - relCueCol] = 0.5
                ## Remove first item in switch
                switch = np.delete(switch, 0)
        dataDict['relCueCol'].append(relCueCol)
        
        # Stimuli
        stim = np.random.choice(n_stim, n_stim, p =  [0.5, 0.5])
        
        dataDict['relCue'].append(stim[relCueCol])
        dataDict['irrelCue'].append(stim[1 - relCueCol])
        
        ##SG: Target is 'randomly' selected between 0 and 1 with the
            # probability of the relevant cue. relCueCol- is to not always
            # have 0 with the highest probability. abs() to prevent the
            # target from being -1.
        target = abs(relCueCol - np.random.choice(
            n_stim, p=[prob[stim[relCueCol]], 1 - prob[stim[relCueCol]]]))
        dataDict['targetLoc'].append(target)
        dataDict['validity'].append(stim[relCueCol] == target)
    
    data = pd.DataFrame(dataDict, columns=column_list)

    return data



#%% ~~ Models (1 cue) ~~ %%#
#--------------------------#


# ~~ Rescorla - Wagner ~~ #
def simRW_1c(parameters, data):
    """
    Rescorla - Wagner learning model with SoftMax
    Policy calculations in model.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is the learning rate in the model.
        Second parameter is constant in action selection method.
        The default is (0.01, 0.5).
    data : pandas.DataFrame
        Experimental structure.

    Returns
    -------
    simData : pandas.DataFrame
        Contains columns of 'data' and:
            0. 'selCue_RW' - selected cue by the Rescorla-Wagner model
            1. 'reward_RW' - reward based on cue selected by Rescorla-Wagner
            2. 'RPE_RW' - reward prediction error
            3. 'Qest_0_RW' - estimated value of cue 0
            4. 'Qest_1_RW' - estimated value of cue 1

    """


    # Variables
    #----------
    # Dataframe
    var_list = [
        'selCue_RW', 'reward_RW',
        'RPE_RW', 'Qest_0_RW', 'Qest_1_RW'
        ]
    simDict = {vari:[] for vari in var_list}
    
    ## Number of trials
    n_trials = data.shape[0]
    ## Number of cues
    N_CUES = 2
    # Model
    ## Estimated value of cue
    Q_est = np.full((n_trials, N_CUES), 1/N_CUES)
    
    # Policy
    ## Selected cues
    selcue = np.nan

    # Trial loop
    for triali, trial in data.iterrows():
        # Policy
        #-------
        ## Random for first trial
        if triali == 0:
            selcue = np.random.randint(N_CUES)
        else:
            ## Probability of cue 0
            probcue = np.exp(parameters[1] * Q_est[triali, 0]) / \
                np.sum(np.exp(np.multiply(parameters[1], Q_est[triali, :])))
            ##SG: If the probability of cue 0 is smaller than a random value,
                # follow cue 1.
            temp = np.random.rand() <= probcue
            ## Action selection
            selcue = temp == 0
        simDict['selCue_RW'].append(selcue)
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward =1, if cue!=target reward = 0
        reward = selcue == trial.targetLoc
        simDict['reward_RW'].append(reward)
        
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
        simDict['RPE_RW'].append(rpe)
        for qi, q_est in enumerate(Q_est[triali, :]):
            simDict[f'Qest_{qi}_RW'].append(q_est)
    
    # Save data
    simData = pd.DataFrame(simDict, columns=var_list)
    simData.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    simData = pd.concat([data, simData], axis=1)

    return simData


# ~~ RW-PH Hybrid ~~ #
def simHybrid_1c(parameters, data, salpha=0.01):
    """
    RW-PH hybrid model with SoftMax
    Policy calculations in model.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is eta.
        Second parameter is constant in action selection method.
        The default is (0.01, 0.5).
    data : pandas.DataFrame
        Experimental structure.
    salpha : TYPE, optional
        Start of alpha. The default is 0.01.

    simData : pandas.DataFrame
        Contains columns of 'data' and:
            0. 'selCue_hyb' - selected cue by the hybrid model
            1. 'reward_hyb' - reward based on cue selected by hybrid model
            2. 'alpha_hyb' - learning rate
            3. 'RPE_hyb' - reward prediction error
            4. 'Qest_0_hyb' - estimated value of cue 0
            5. 'Qest_1_hyb' - estimated value of cue 1

    """


    # Variables
    #----------
    # Dataframe
    var_list = [
        'selCue_hyb', 'reward_hyb', 'alpha_hyb'
        'RPE_hyb', 'Qest_0_hyb', 'Qest_1_hyb'
        ]
    simDict = {vari:[] for vari in var_list}
    
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
    ## Selected cues
    selcue = np.nan

    # Trial loop
    for triali, trial in data.iterrows():
        # Select cue
        ## Random for first trial
        if triali == 0:
            selcue = np.random.randint(N_CUES)
        else:
            ## Probability of cue 0
            probcue = np.exp(parameters[1] * Q_est[triali, 0]) / \
                np.sum(np.exp(np.multiply(parameters[1], Q_est[triali, :])))
            ##SG: If the probability of cue 0 is smaller than a random value,
                # follow cue 1.
            temp = np.random.rand() <= probcue
            ## Action selection
            selcue = temp == 0
        simDict['selCue_hyb']
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward =1, if cue!=target reward = 0
        reward = selcue == trial.targetLoc
        
        # Hybrid
        #-------
        if triali == 0:
            # Reward prediction error
            rpe = reward - Q_est[triali, selcue]
            # Alpha (PH)
            alpha[triali, selcue] = parameters[0] * np.abs(rpe) + \
                (1 - parameters[0]) * alpha[triali, selcue]
            # Cue estimates
            Q_est[triali, selcue] = Q_est[triali, selcue] + alpha[triali, selcue] * rpe
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
            Q_est[triali, selcue] = Q_est[triali, selcue] + alpha[triali, selcue] * rpe
        simDict['RPE_hyb'].append(rpe)
        simDict['alpha_hyb'].append(alpha[triali, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            simDict[f'Qest_{qi}_hyb'].append(q_est)
    
    # Save data
    simData = pd.DataFrame(simDict, columns=var_list)
    simData.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    simData = pd.concat([data, simData], axis=1)

    return simData



#%% ~~ Models (2 cues) ~~ %%#
#---------------------------#


# ~~ Rescorla - Wagner ~~ #
def simRW_2c(parameters, data):
    """
    Rescorla - Wagner learning model with SoftMax
    Both cue estimates are being updated.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is the learning rate in the model.
        Second parameter is constant in action selection method.
        The default is (0.01, 0.5).
    data : pandas.DataFrame
        Experimental structure.

    Returns
    -------
    simData : pandas.DataFrame
        Contains columns of 'data' and:
            0. 'selCue_RW' - selected cue by the Rescorla-Wagner model
            1. 'reward_RW' - reward based on cue selected by Rescorla-Wagner
            2. 'RPE_RW' - reward prediction error
            3. 'Qest_0_RW' - estimated value of cue 0
            4. 'Qest_1_RW' - estimated value of cue 1

    """


    # Variables
    #----------
    # Dataframe
    var_list = [
        'selCue_RW', 'reward_RW',
        'RPE_RW', 'Qest_0_RW', 'Qest_1_RW'
        ]
    simDict = {vari:[] for vari in var_list}
    
    ## Number of trials
    n_trials = data.shape[0]
    ## Number of cues
    N_CUES = 2
    # Model
    ## Estimated value of cue
    Q_est = np.full((n_trials, N_CUES), 1/N_CUES)
    
    # Policy
    ## Selected cues
    selcue = np.nan

    # Trial loop
    for triali, trial in data.iterrows():
        # Policy
        #-------
        ## Random for first trial
        if triali == 0:
            selcue = np.random.randint(N_CUES)
        else:
            ## Probability of cue 0
            probcue = np.exp(parameters[1] * Q_est[triali, 0]) / \
                np.sum(np.exp(np.multiply(parameters[1], Q_est[triali, :])))
            ##SG: If the probability of cue 0 is smaller than a random value,
                # follow cue 1.
            temp = np.random.rand() <= probcue
            ## Action selection
            selcue = temp == 0
        simDict['selCue_RW'].append(selcue)
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward =1, if cue!=target reward = 0
        reward = selcue == trial.targetLoc
        simDict['reward_RW'].append(reward)
        
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
        simDict['RPE_RW'].append(reward - Q_est[triali - 1, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            simDict[f'Qest_{qi}_RW'].append(q_est)
    
    # Save data
    simData = pd.DataFrame(simDict, columns=var_list)
    simData.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    simData = pd.concat([data, simData], axis=1)

    return simData


# ~~ RW-PH Hybrid ~~ #
def sulHybrid_2c(parameters, data, salpha=0.01):
    """
    RW-PH hybrid model with SoftMax
    Both cue estimates are being updated.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is eta.
        Second parameter is constant in action selection method.
        The default is (0.01, 0.5).
    data : pandas.DataFrame
        Experimental structure.
    salpha : TYPE, optional
        Start of alpha. The default is 0.01.

    simData : pandas.DataFrame
        Contains columns of 'data' and:
            0. 'selCue_hyb' - selected cue by the hybrid model
            1. 'reward_hyb' - reward based on cue selected by hybrid model
            2. 'alpha_hyb' - learning rate
            3. 'RPE_hyb' - reward prediction error
            4. 'Qest_0_hyb' - estimated value of cue 0
            5. 'Qest_1_hyb' - estimated value of cue 1

    """


    # Variables
    #----------
    # Dataframe
    var_list = [
        'selCue_hyb', 'reward_hyb', 'alpha_hyb'
        'RPE_hyb', 'Qest_0_hyb', 'Qest_1_hyb'
        ]
    simDict = {vari:[] for vari in var_list}
    
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
    ## Selected cues
    selcue = np.nan

    # Trial loop
    for triali, trial in data.iterrows():
        # Select cue
        ## Random for first trial
        if triali == 0:
            selcue = np.random.randint(N_CUES)
        else:
            ## Probability of cue 0
            probcue = np.exp(parameters[1] * Q_est[triali, 0]) / \
                np.sum(np.exp(np.multiply(parameters[1], Q_est[triali, :])))
            ##SG: If the probability of cue 0 is smaller than a random value,
                # follow cue 1.
            temp = np.random.rand() <= probcue
            ## Action selection
            selcue = temp == 0
        simDict['selCue_hyb']
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward =1, if cue!=target reward = 0
        reward = selcue == trial.targetLoc
        
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
                Q_est[triali, cuei] = Q_est[triali, cuei] + alpha[triali, cuei] * rpe
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
                Q_est[triali, cuei] = Q_est[triali, cuei] + alpha[triali, cuei] * rpe
        simDict['RPE_hyb'].append(reward - Q_est[triali - 1, selcue])
        simDict['alpha_hyb'].append(alpha[triali, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            simDict[f'Qest_{qi}_hyb'].append(q_est)
    
    # Save data
    simData = pd.DataFrame(simDict, columns=var_list)
    simData.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    simData = pd.concat([data, simData], axis=1)

    return simData



#------------------------------------------------------------------------- End

