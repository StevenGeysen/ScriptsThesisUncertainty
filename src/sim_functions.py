#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Simulation functions -- Version 2.2.1
Last edit:  2022/07/05
Author(s):  Geysen, Steven (SG)
Notes:      - Functions used for the simulation of the task used by
                Marzecova et al. (2019). Both structure and models.
                * Models
                    - Rescorla-Wagner (Daphne)
                    - Rescorla-Wagner - Pearce-Hall hybrid (Hugo)
                    - Win-stay-lose-shift (Wilhelm)
                    - Random (Renee)
                * (negaitve) log likelihood
                * Negative Spearman correlation
            - Release notes:
                * Policy function with argmax and SoftMax in af
            
To do:      - 
            
Comments:   SG: Simulations return pandas.DataFrame. Model functions return
                originale data with model selection appended.
            
Sources:    https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00612/full
            https://docs.sympy.org/latest/modules/stats.html
            https://github.com/Kingsford-Group/ribodisomepipeline/blob/master/scripts/exGaussian.py
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html
            https://hannekedenouden.ruhosting.nl/RLtutorial/html/SoftMax.html
            https://lindeloev.shinyapps.io/shiny-rt/
"""



#%% ~~ Imports ~~ %%#



import numpy as np
import pandas as pd

import src.assisting_functions as af

from scipy import stats



#%% ~~ Structure ~~ %%#
#######################


def sim_experiment(simnr=1, ntrials=640, nswitch=7):
    """
    Simulate the eperimental structure of Marzecova et al. (2019)

    Parameters
    ----------
    simnr : int, optional
        Simulation number. Used for the randomisation of the reward
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
    dataDict['id'] = simnr
    
    # Task
    n_stim = 2
    ## Colour of relevant cue
    relCueCol = np.random.randint(n_stim)
    ## Probabilities of reward for each stim
    prob = np.full(n_stim, 0.5)
    ## SG: For simulations with an even number is the initial probability
        ## of the relevant cue 0.7
    if simnr % 2 == 0:
        prob[relCueCol] = 0.7
    else:
        prob[relCueCol] = 0.85
    ## Trials where probability is switched
    switch = np.cumsum(np.random.randint(40, 120, size = nswitch))

    # Trial loop
    #-----------
    for triali in range(ntrials):
        dataDict['trial'].append(triali)
        # Switch probabilities
        if switch.shape[0] != 0:
            if triali == switch[0]:
                relCueCol = 1 - relCueCol
                
                if simnr % 2 == 0:
                    ##SG: For simulations with an even number is in the
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



#%% ~~ Blocks ~~ %%#
####################


def sim_rt(selcue, probcue):
    """
    Simulate response times
    Response times, sampled from ExGaussian distribution. Sigma and mu are
    taken from exgauss fit of the original data. Tau is the probability of the
    selected cue, used as proxy for difficulty.

    Parameters
    ----------
    selcue : int
        Selected cue.
    probcue : float
        Probability that cue O is selected.

    Returns
    -------
    RT : float
        Simulated response time.
    """

    try:
        ##SG: K = tau / sigma, loc = mu, scale = sigma.
        RT = stats.exponnorm.rvs(K = abs(selcue - probcue) / 0.02635,
                                 loc = 0.3009, scale = 0.02635)
    except:
        RT = np.nan
        print('Failed rt sampling')
    if RT == float('inf'):
        RT = 1.7

    return RT


def sim_models():
    """
    Simulation dictionary
    Contains the functions of the different simulation models.
    """

    return {'RW': simRW_1c,
            'RW2': simRW_2c,
            'H': simHybrid_1c,
            'H2': simHybrid_2c,
            'W': simWSLS,
            'R': simRandom}



#%% ~~ Models ~~ %%#
####################


#%% ~~ Models (1 cue) ~~ %%#
#--------------------------#


# ~~ Rescorla - Wagner ~~ #
def simRW_1c(parameters, data, asm='soft'):
    """
    Daphne the delta learner
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
    asm : string, optional
        The action selection method, policy. The default is SoftMax.

    Returns
    -------
    simData : pandas.DataFrame
        Contains columns of 'data' and simulated behaviour of Daphne:
            0. 'selCue_RW' - selected cue by the Rescorla-Wagner model
            1. 'prob_RW' - probability to select cue 0
            2. 'rt_RW' - response time
            3. 'reward_RW' - reward based on cue selected by Rescorla-Wagner
            4. 'RPE_RW' - reward prediction error
            5. 'Qest_0_RW' - estimated value of cue 0
            6. 'Qest_1_RW' - estimated value of cue 1
    """

    # Variables
    #----------
    # Dataframe
    var_list = [
        'selCue_RW', 'prob_RW', 'rt_RW', 'reward_RW',
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
            selcue, probcue = af.policy(asm, Q_est[triali - 1, :],
                                        parameters[1])
        simDict['selCue_RW'].append(selcue)
        simDict['prob_RW'].append(probcue)
        
        # Response time
        simDict['rt_RW'].append(sim_rt(selcue, probcue))
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcue == trial.targetLoc)
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

    return af.save_data(simDict, data, var_list)


# ~~ RW-PH Hybrid ~~ #
def simHybrid_1c(parameters, data, salpha=0.01, asm='soft'):
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
    simData : pandas.DataFrame
        Contains columns of 'data' and simulated behaviour of Hugo:
            0. 'selCue_H' - selected cue by the hybrid model
            1. 'prob_H' - probability to select cue 0
            2. 'rt_H' - response time
            3. 'reward_H' - reward based on cue selected by hybrid model
            4. 'alpha_H' - learning rate
            5. 'RPE_H' - reward prediction error
            6. 'Qest_0_H' - estimated value of cue 0
            7. 'Qest_1_H' - estimated value of cue 1
    """

    # Variables
    #----------
    # Dataframe
    var_list = [
        'selCue_H', 'prob_H', 'rt_H', 'reward_H', 'alpha_H',
        'RPE_H', 'Qest_0_H', 'Qest_1_H'
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
            selcue, probcue = af.policy(asm, Q_est[triali - 1, :],
                                        parameters[1])
        simDict['selCue_H'].append(selcue)
        simDict['prob_H'].append(probcue)
        
        # Response time
        simDict['rt_H'].append(sim_rt(selcue, probcue))
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcue == trial.targetLoc)
        simDict['reward_H'].append(reward)
        
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
        simDict['RPE_H'].append(rpe)
        simDict['alpha_H'].append(alpha[triali, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            simDict[f'Qest_{qi}_H'].append(q_est)

    return af.save_data(simDict, data, var_list)



#%% ~~ Models (2 cues) ~~ %%#
#---------------------------#


# ~~ Rescorla - Wagner ~~ #
def simRW_2c(parameters, data, asm='soft'):
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
            2. 'rt_RW2' - response time
            3. 'reward_RW2' - reward based on cue selected by Rescorla-Wagner
            4. 'RPE_RW2' - reward prediction error
            5. 'Qest_0_RW2' - estimated value of cue 0
            6. 'Qest_1_RW2' - estimated value of cue 1
    """

    # Variables
    #----------
    # Dataframe
    var_list = [
        'selCue_RW2', 'prob_RW2', 'rt_RW2', 'reward_RW2',
        'RPE_RW2', 'Qest_0_RW2', 'Qest_1_RW2'
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
            selcue, probcue = af.policy(asm, Q_est[triali - 1, :],
                                        parameters[1])
        simDict['selCue_RW2'].append(selcue)
        simDict['prob_RW2'].append(probcue)
        
        # Response time
        simDict['rt_RW2'].append(sim_rt(selcue, probcue))
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcue == trial.targetLoc)
        simDict['reward_RW2'].append(reward)
        
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
        simDict['RPE_RW2'].append(reward - Q_est[triali - 1, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            simDict[f'Qest_{qi}_RW2'].append(q_est)

    return af.save_data(simDict, data, var_list)


# ~~ RW-PH Hybrid ~~ #
def simHybrid_2c(parameters, data, salpha=0.01, asm='soft'):
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
            2. 'rt_H2' - response time
            3. 'reward_H2' - reward based on cue selected by hybrid model
            4. 'alpha_H2' - learning rate
            5. 'RPE_H2' - reward prediction error
            6. 'Qest_0_H2' - estimated value of cue 0
            7. 'Qest_1_H2' - estimated value of cue 1
    """

    # Variables
    #----------
    # Dataframe
    var_list = [
        'selCue_H2', 'prob_H2', 'rt_H2', 'reward_H2', 'alpha_H2',
        'RPE_H2', 'Qest_0_H2', 'Qest_1_H2'
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
            selcue, probcue = af.policy(asm, Q_est[triali - 1, :],
                                        parameters[1])
        simDict['selCue_H2'].append(selcue)
        simDict['prob_H2'].append(probcue)
        
        # Response time
        simDict['rt_H2'].append(sim_rt(selcue, probcue))
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        reward = int(selcue == trial.targetLoc)
        simDict['reward_H2'].append(reward)
        
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
        simDict['RPE_H2'].append(reward - Q_est[triali - 1, selcue])
        simDict['alpha_H2'].append(alpha[triali, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            simDict[f'Qest_{qi}_H2'].append(q_est)

    return af.save_data(simDict, data, var_list)



#%% ~~ Controle models ~~ %%#
#---------------------------#


# ~~ Wilhelm ~~ #
def simWSLS(data):
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
    simDict = {vari:[] for vari in var_list}
    
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
        
        simDict['selCue_W'].append(selcues[triali])
        ##SG: If there is some uncertainty in which cue Wilhelm will pick,
            # there is something wrong in the model.
        simDict['prob_W'].append(1)
        simDict['reward_W'].append(reward)

    return af.save_data(simDict, data, var_list)


# ~~ Renee ~~ #
def simRandom(data):
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
    simDict = {vari:[] for vari in var_list}
    
    # Number of cues
    N_CUES = 2

    # Trial loop
    for triali, trial in data.iterrows():
        selcue = np.random.randint(N_CUES)
        simDict['selCue_R'].append(selcue)
        simDict['prob_R'].append(0.5)
        reward = int(selcue == trial.targetLoc)
        simDict['reward_R'].append(reward)

    return af.save_data(simDict, data, var_list)



#%% ~~ Others ~~ %%#
####################


def sim_negLL(thetas, data, model):
    """
    Negative log-likelihood of choice based on RW predictions of simulations

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
    Negative log-likelihood of selected stimuli.
    """

    # Rename to avoid duplicates
    data = data.rename(columns={f'selCue_{model}':'selCue',
                                f'prob_{model}':'prob'})
    # Log-likelihood of choice
    loglik_of_choice = []
    # Simulate data
    modelDict = sim_models()
    if not model.upper() in ['W', 'R']:
        simData = modelDict[model.upper()](thetas, data)
    else:
        simData = modelDict[model.upper()](data)
    
    for _, trial in simData.iterrows():
        # Model estimated value of model's selected stimulus
        ##SG: Likelihood of left stimulus if model picked left.
            # Otherwise 1-left for likelihood of right stimulus.
        picked_prob = abs(trial[f'selCue_{model.upper()}'] - 
                          trial[f'prob_{model}'])
        loglik_of_choice.append(np.log(picked_prob))

    # Return negative log likelihood
    # return -1 * np.nansum(loglik_of_choice)
    return np.nansum(loglik_of_choice)


def sim_negSpearCor(thetas, data, model):
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

    # Wilhelm and Renee do not simulate RTs.
    assert not model.upper() in ['W', 'R'], 'Model has no simulated RT'

    # Rename to avoid duplicates
    data = data.rename(columns={f'selCue_{model}':'selCue',
                                f'prob_{model}':'prob'})
    # Simulate data
    modelDict = sim_models()
    simData = modelDict[model.upper()](thetas, data)

    # Correlation between RT and RPE
    return - stats.spearmanr(simData[f'rt_{model.upper()}'],
                             simData[f'RPE_{model.upper()}'],
                             nan_policy = 'omit')[0]



#------------------------------------------------------------------------- End
