#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Simulation functions -- Version 4.2
Last edit:  2022/09/24
Author(s):  Geysen, Steven (SG)
Notes:      - Functions used for the simulation of the task used by
                Marzecova et al. (2019). Both structure and models.
                * Models
                    - Rescorla-Wagner (Daphne)
                    - Rescorla-Wagner - Pearce-Hall hybrid (Hugo)
                    - Meta learner (Michelle)
                    - Win-stay-lose-shift (Wilhelm)
                    - Random (Renee)
                * (negaitve) log likelihood
                * Negative Spearman correlation
                * Accuracy
                * var_bin_switch
            - Release notes:
                * Changed reward calculations
                * Target in sim_experiment changed
                * Valitdity corrected
                * Negative correlation with absolute values
            
To do:      - Meta learner
            
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

import fns.assisting_functions as af

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
            7. 'gammaBlock' - the validity level within the block
    """

    # Variables
    #----------
    # DataFrame
    column_list = [
        'id', 'trial', 'relCueCol', 'relCue', 'irrelCue', 'targetLoc',
        'validity', 'gammaBlock'
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
        dataDict['gammaBlock'].append(prob[relCueCol])
        
        # Stimuli
        stim = np.random.choice(n_stim, n_stim, p =  [0.5, 0.5])
        
        relcue = stim[relCueCol]
        irrelcue = stim[1 - relCueCol]
        dataDict['relCue'].append(relcue)
        dataDict['irrelCue'].append(irrelcue)
        
        ##SG: Target is 'randomly' selected between 0 and 1 with the
            # probability of the relevant cue.
        target = np.random.choice(
            [relcue, 1 - relcue], p = [prob[relCueCol], 1 -  prob[relCueCol]]
            )
        dataDict['targetLoc'].append(target)
        ##SG: Valid trials are coded as 0, invalid as 1. For consistancy with
            # behavioural data. Now normal
        dataDict['validity'].append(int(relcue == target))
    
    data = pd.DataFrame(dataDict, columns=column_list)

    return data



#%% ~~ Blocks ~~ %%#
####################


def sim_rt(rpe):
    """
    Simulate response times
    Response times, sampled from ExGaussian distribution. Sigma and mu are
    taken from exgauss fit of the original data. Tau is the reward prediction
    error of the selected cue, used as proxy for difficulty (absolute values to
    avoid errors caused by negative values).

    Parameters
    ----------
    rpe : int
        Reward prediction error of the selected cue.

    Returns
    -------
    RT : float
        Simulated response time (in secondes).
    """

    if rpe == 0:
        RT = 0.3
    else:
        try:
            ##SG: K = tau / sigma, loc = mu, scale = sigma.
            RT = stats.exponnorm.rvs(K = abs(rpe) / 0.02635,
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
            'M': simMeta_1c,
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
        First parameter is the learning rate of the model (0 <= alpha <= 1).
        Second parameter is the constant of the action selection method
            (0 < beta)..
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
            3. 'reward_RW' - reward based on cue selected by Rescorla-Wagner,
                also the validity of the selected cue
            4. 'RPE_RW' - reward prediction error
            5. 'Men_PE_RW' - Prediction error calculated as in
                Mengotti et al. (2017)
            6. 'Qest_0_RW' - estimated value of cue 0
            7. 'Qest_1_RW' - estimated value of cue 1
    """

    # Variables
    # ---------
    # Dataframe
    var_list = [
        'selCue_RW', 'prob_RW', 'rt_RW', 'reward_RW',
        'RPE_RW', 'Men_PE_RW', 'Qest_0_RW', 'Qest_1_RW'
        ]
    simDict = {vari:[] for vari in var_list}
    
    ## Number of trials
    n_trials = data.shape[0]
    ## Number of cues
    N_CUES = 2
    # Model
    ## Estimated value of cue
    Q_est = np.full((n_trials, N_CUES), 1/N_CUES)
    ## Outcomes
    outcomes = np.full((2, ), np.nan)
    
    # Policy
    ## Selected cue
    selcue = np.nan
    
    # Parameters
    alpha = parameters[0]
    beta, bias = 0, 0
    if len(parameters) > 1:
        beta = parameters[1]
    if len(parameters) > 2:
        bias = parameters[2]

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
                                        beta, bias)
        simDict['selCue_RW'].append(selcue)
        simDict['prob_RW'].append(probcue)
        
        # Reward
        # ------
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        """
        AM: The model chooses the relevant cue color (1 - white, 2 - black),
        but you need to also know the direction of the cue
        (1 - left,  2 - right). The direction of the cue is in the column
        relCue or irrelCue, but the relCue or irrelCue refer to the gorund
        thruth and not to what the model chooses (selCue). RelCue and irrelCue
        don’t give you the information about which color is the relevant Cue
        currently, this information is in the relCueCol (1-white 2 -black). So,
        it would be easier to recode the data from the perspective of whiteCue
        and blackCue: if white cue is relevant, you would put the value for
        relCue direction in the column for white cue, while you would put the
        value for direction of the irelCue for the black cue.
        """
        outcomes[int(trial.relCueCol)] = trial.relCue == trial.targetLoc
        outcomes[int(1 - trial.relCueCol)] = trial.irrelCue == trial.targetLoc
        reward = int(outcomes[selcue])
        simDict['reward_RW'].append(reward)
        
        # Update rule (RW)
        # ----------------
        if triali == 0:
            # Reward prediction error
            rpe = reward - Q_est[triali, selcue]
        else:
            # Reward prediction error
            rpe = reward - Q_est[triali - 1, selcue]
            # Cue estimates
            ## Repeat cue estimates of previous trial
            Q_est[triali, :] = Q_est[triali - 1, :]
        ## Update cue estimate of selected stimulus in current trial
        Q_est[triali, selcue] = Q_est[triali, selcue] + alpha * rpe
        simDict['RPE_RW'].append(rpe)
        simDict['Men_PE_RW'].append(
            1 - reward + Q_est[triali - 1, selcue] * (2 * reward - 1)
            )
        for qi, q_est in enumerate(Q_est[triali, :]):
            simDict[f'Qest_{qi}_RW'].append(q_est)
        
        # Response time
        simDict['rt_RW'].append(sim_rt(rpe))

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
        First parameter is eta (0 <= eta <= 1).
        Second parameter is the constant of the action selection method
            (0 < beta).
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
    # ---------
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
    ## Outcomes
    outcomes = np.full((2, ), np.nan)
    
    # Policy
    ## Selected cue
    selcue = np.nan
    
    # Parameters
    eta = parameters[0]
    beta, bias = 0, 0
    if len(parameters) > 1:
        beta = parameters[1]
    if len(parameters) > 2:
        bias = parameters[2]

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
                                        beta, bias)
        simDict['selCue_H'].append(selcue)
        simDict['prob_H'].append(probcue)
        
        # Reward
        # ------
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        outcomes[int(trial.relCueCol)] = trial.relCue == trial.targetLoc
        outcomes[int(1 - trial.relCueCol)] = trial.irrelCue == trial.targetLoc
        reward = int(outcomes[selcue])
        simDict['reward_H'].append(reward)
        
        # Hybrid
        # ------
        if triali == 0:
            # Reward prediction error
            rpe = reward - Q_est[triali, selcue]
        else:
            # Reward prediction error
            rpe = reward - Q_est[triali - 1, selcue]
            # Update values of selected stimulus in current trial
            alpha[triali, :] = alpha[triali - 1, :]
            Q_est[triali, :] = Q_est[triali - 1, :]
        # Alpha (PH)
        alpha[triali, selcue] = eta * np.abs(rpe) + \
            (1 - eta) * alpha[triali, selcue]
        # Cue estimates
        Q_est[triali, selcue] = Q_est[triali, selcue] + \
            alpha[triali, selcue] * rpe
        simDict['RPE_H'].append(rpe)
        simDict['alpha_H'].append(alpha[triali, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            simDict[f'Qest_{qi}_H'].append(q_est)
        
        # Response time
        simDict['rt_H'].append(sim_rt(rpe))

    return af.save_data(simDict, data, var_list)


def simMeta_1c(parameters, data, asm='soft'):
    """
    Michelle the meta learner
    The meta learning model of Cohen and his team.
    https://doi.org/10.1016/j.neuron.2019.06.001
    https://doi.org/10.1016/j.cub.2021.12.006

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is the learning rate of the model (0 <= alpha <= 1).
        Second parameter is the forgetting rate of the model (zeta).
        Third parameter is the constant of the action selection method
            (0 < beta).
        Fourth parameter is the bias term of the biased SoftMax
            (-1 (cue_0 bias) <= bias <= 1 (cue_1 bias)). It indicates a prefere
            for one cue over the other. The default is 0 (no bias).
    data : pandas.DataFrame
        Dataframe containing the structure of the experiment.
    asm : string, optional
        The action selection method, policy. The default is SoftMax.

    Returns
    -------
    simData : pandas.DataFrame
        Contains columns of 'data' and simulated behaviour of Michelle:
            0. 'selCue_M' - selected cue by the meta learning model
            1. 'prob_M' - probability to select cue 0
            2. 'rt_M' - response time
            3. 'reward_M' - reward based on cue selected by meta learner
            4. 'RPE_M' - reward prediction error
            5. 'Qest_0_M' - estimated value of cue 0
            6. 'Qest_1_M' - estimated value of cue 1
            7. 'eun_0_M' - estimate of expected uncertainty of cue 0
            8. 'eun_1_M' - estimate of expected uncertainty of cue 1
            9. 'uun_M' - unexpected uncertainty
    """
    # Variables
    # ---------
    # Dataframe
    var_list = [
        'selCue_M', 'prob_M', 'rt_M', 'reward_M',
        'RPE_M', 'Qest_0_M', 'Qest_1_M',
        'eun_0_M', 'eun_1_M', 'uun_M'
        ]
    simDict = {vari:[] for vari in var_list}
    
    # Number of trials
    n_trials = data.shape[0]
    # Number of cues
    N_CUES = 2
    # Model
    ## Estimated value of cue
    Q_est = np.full((n_trials, N_CUES), 1/N_CUES)
    ## Outcomes
    outcomes = np.full((2, ), np.nan)
    ## Estimate of expected uncertainty calculated from the history of URPEs
    eun = np.full((n_trials, ), 1/N_CUES)
    ## Unexpected uncertainty
    uun = 0
    
    # Policy
    ## Selected cue
    selcue = np.nan
    
    # Parameters
    alpha, zeta = parameters[:2]
    beta, bias = 0, 0
    if len(parameters) > 2:
        beta = parameters[2]
    if len(parameters) > 3:
        bias = parameters[3]

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
                                        beta, bias)
        simDict['selCue_M'].append(selcue)
        simDict['prob_M'].append(probcue)
        
        # Reward
        # ------
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        outcomes[int(trial.relCueCol)] = trial.relCue == trial.targetLoc
        outcomes[int(1 - trial.relCueCol)] = trial.irrelCue == trial.targetLoc
        reward = int(outcomes[selcue])
        simDict['reward_M'].append(reward)
        
        # Update rule
        # -----------
        if triali == 0:
            # Reward prediction error
            rpe = reward - Q_est[triali, selcue]
        else:
            # Reward prediction error
            rpe = reward - Q_est[triali - 1, selcue]
            # Cue estimates
            ## Repeat values of previous trial
            Q_est[triali, :] = Q_est[triali - 1, :]
            eun[trial] = eun[trial - 1]
        ## Update expected and unexpected uncertainty
        uun = abs(rpe) - eun[triali]
        eun[triali] *= uun
        ## Update cue estimate of selected stimulus in current trial
        Q_est[triali, selcue] = Q_est[triali, selcue] + \
            alpha * rpe * (1 - eun[triali])
        ## Forget not selected stimulus
        Q_est[triali, abs(1 - selcue)] = zeta * Q_est[triali, abs(1 - selcue)]
        simDict['RPE_M'].append(rpe)
        for qi, q_est in enumerate(Q_est[triali, :]):
            simDict[f'Qest_{qi}_M'].append(q_est)
        
        # Response time
        simDict['rt_G'].append(sim_rt(rpe))

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
        First parameter is the learning rate of the model (0 <= alpha <= 1).
        Second parameter is the constant of the action selection method
            (0 < beta).
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
    # ---------
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
    ## Outcomes
    outcomes = np.full((2, ), np.nan)
    
    # Policy
    ## Selected cue
    selcue = np.nan
    
    # Parameters
    alpha = parameters[0]
    beta, bias = 0, 0
    if len(parameters) > 1:
        beta = parameters[1]
    if len(parameters) > 2:
        bias = parameters[2]

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
                                        beta, bias)
        simDict['selCue_RW2'].append(selcue)
        simDict['prob_RW2'].append(probcue)
        
        # Reward
        # ------
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        outcomes[int(trial.relCueCol)] = trial.relCue == trial.targetLoc
        outcomes[int(1 - trial.relCueCol)] = trial.irrelCue == trial.targetLoc
        reward = int(outcomes[selcue])
        simDict['reward_RW2'].append(reward)
        
        # Update rule (RW)
        # ----------------
        if triali == 0:
            for cuei in range(N_CUES):
                # Reward prediction error
                rpe = reward - Q_est[triali, cuei]
                # Cue estimates
                Q_est[triali, cuei] = Q_est[triali, cuei] + alpha * rpe
        else:
            # Repeat cue estimates of previous trial
            Q_est[triali, :] = Q_est[triali - 1, :]
            for cuei in range(N_CUES):
                # Reward prediction error
                rpe = reward - Q_est[triali - 1, cuei]
                # Cue estimates
                Q_est[triali, cuei] = Q_est[triali, cuei] + alpha * rpe
        simDict['RPE_RW2'].append(reward - Q_est[triali - 1, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            simDict[f'Qest_{qi}_RW2'].append(q_est)
        
        # Response time
        simDict['rt_RW2'].append(sim_rt(rpe))

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
        First parameter is eta (0 <= eta <= 1).
        Second parameter is the constant of the action selection method
            (0 < beta).
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
    ## Outcomes
    outcomes = np.full((2, ), np.nan)
    
    # Policy
    ## Selected cue
    selcue = np.nan
    
    # Parameters
    eta = parameters[0]
    beta, bias = 0, 0
    if len(parameters) > 1:
        beta = parameters[1]
    if len(parameters) > 2:
        bias = parameters[2]

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
                                        beta, bias)
        simDict['selCue_H2'].append(selcue)
        simDict['prob_H2'].append(probcue)
        
        # Reward
        # ------
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        outcomes[int(trial.relCueCol)] = trial.relCue == trial.targetLoc
        outcomes[int(1 - trial.relCueCol)] = trial.irrelCue == trial.targetLoc
        reward = int(outcomes[selcue])
        simDict['reward_H2'].append(reward)
        
        # Hybrid
        # ------
        if triali == 0:
            for cuei in range(N_CUES):
                # Reward prediction error
                rpe = reward - Q_est[triali, cuei]
                # Alpha (PH)
                alpha[triali, cuei] = eta * np.abs(rpe) + \
                    (1 - eta) * alpha[triali, cuei]
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
                alpha[triali, cuei] = eta * np.abs(rpe) + \
                    (1 - eta) * alpha[triali, cuei]
                # Cue estimates
                Q_est[triali, cuei] = Q_est[triali, cuei] + \
                    alpha[triali, cuei] * rpe
        simDict['RPE_H2'].append(reward - Q_est[triali - 1, selcue])
        simDict['alpha_H2'].append(alpha[triali, selcue])
        for qi, q_est in enumerate(Q_est[triali, :]):
            simDict[f'Qest_{qi}_H2'].append(q_est)
        
        # Response time
        simDict['rt_H2'].append(sim_rt(rpe))

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
    ## Outcomes
    outcomes = np.full((2, ), np.nan)
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
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward = 1, if cue!=target reward = 0
        outcomes[int(trial.relCueCol)] = trial.relCue == trial.targetLoc
        outcomes[int(1 - trial.relCueCol)] = trial.irrelCue == trial.targetLoc
        reward = int(outcomes[int(selcues[triali])])
        
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
    ## Outcomes
    outcomes = np.full((2, ), np.nan)

    # Trial loop
    for triali, trial in data.iterrows():
        selcue = np.random.randint(N_CUES)
        simDict['selCue_R'].append(selcue)
        simDict['prob_R'].append(0.5)
        outcomes[int(trial.relCueCol)] = trial.relCue == trial.targetLoc
        outcomes[int(1 - trial.relCueCol)] = trial.irrelCue == trial.targetLoc
        reward = int(outcomes[selcue])
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

    model = model.upper()
    # Rename to avoid duplicates
    data = data.rename(columns={f'selCue_{model}': 'selCue',
                                f'prob_{model}': 'prob'})
    # Log-likelihood of choice
    loglik_of_choice = []
    # Simulate data
    modelDict = sim_models()
    if not model in ['W', 'R']:
        simData = modelDict[model](thetas, data)
    else:
        simData = modelDict[model](data)
    
    for _, trial in simData.iterrows():
        # Model estimated value of model's selected stimulus
        ##SG: Likelihood of left stimulus if model picked left.
            # Otherwise 1-left for likelihood of right stimulus.
        picked_prob = abs(trial[f'selCue_{model}'] -
                          trial[f'prob_{model}'])
        loglik_of_choice.append(np.log(picked_prob))

    # Return negative log likelihood
    # return -1 * np.nansum(loglik_of_choice)
    return np.nansum(loglik_of_choice)


def sim_negSpearCor(thetas, data, model, asm='soft'):
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
            M - Meta learner
    asm : string, optional
        The action selection method, policy. The default is SoftMax.

    Returns
    -------
    negative spearman r
    """

    model = model.upper()
    # Wilhelm and Renee do not simulate RTs.
    assert not model in ['W', 'R'], 'Model has no simulated RT'

    # Rename to avoid duplicates
    data = data.rename(columns={f'selCue_{model}': 'selCue',
                                f'prob_{model}': 'prob',
                                f'rt_{model}': 'rt',
                                f'RPE_{model}': 'RPE'})
    # Simulate data
    modelDict = sim_models()
    simData = modelDict[model](parameters=thetas, data=data, asm=asm)

    # Correlation between RT and RPE
    return - stats.spearmanr(simData[f'rt_{model}'].to_numpy(),
                             # simData[f'RPE_{model}'].to_numpy(),
                             abs(simData[f'RPE_{model}']).to_numpy(),
                             nan_policy = 'omit')[0]


def sim_accuracy(thetas, data, model, asm='soft'):
    """
    Model accuracy

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
            M - Meta learner
            W - Win-stay-lose-shift
            R - Random
    asm : string, optional
        The action selection method, policy. The default is SoftMax.

    Returns
    -------
    Accuracy
    """

    # Rename to avoid duplicates
    data = data.rename(columns={f'selCue_{model}': 'selCue',
                                f'prob_{model}': 'prob',
                                f'rt_{model}': 'rt',
                                f'RPE_{model}': 'RPE'})
    
    model = model.upper()
    # Simulate data
    modelDict = sim_models()
    if model in ['W', 'R']:
        simData = modelDict[model](data)
    else:
        simData = modelDict[model](parameters=thetas, data=data, asm=asm)

    return sum(simData['relCue'] == simData[f'selCue_{model}']) / len(data)


def var_bin_switch(
        dataList, datadir, varList, validity='validity', bin_size=15, uun='All'
        ):
    """
    Bin variability effect of simulated data before and after switch

    Parameters
    ----------
    dataList : list
        List containing the data filenames.
    datadir : Path
        Location of the data.
    varList : tuple, list, array
        List with the names of the relevant variables.
    bin_size : int, optional
        Number of trials grouped together. The default is 15.

    Returns
    -------
    post15 : dictionary
        First 15 trials after switch.
    middle15 : dictionary
        Trials 15 to 30.
    leftover : dictionary
        All trials except for first 30.
    last15 : dictionary
        Last 15 trials, less if there are less then 45 trials between switches.
    pre15 : dictionary
        The 15 trials before switch.
    """

    # Trial bins
    post15 = {vari:[] for vari in varList}
    middle15 = {vari:[] for vari in varList}
    leftover = {vari:[] for vari in varList}
    last15 = {vari:[] for vari in varList}
    pre15 = {vari:[] for vari in varList}

    for simi, filei in enumerate(dataList):
        sim_data = pd.read_csv(datadir / filei, index_col='Unnamed: 0')
        if uun.upper() == 'HIGH':
            sim_data = sim_data[sim_data['gammaBlock'] <= 0.8]
        elif uun.upper() == 'LOW':
            sim_data = sim_data[sim_data['gammaBlock'] > 0.8]
        sim_data.reset_index(drop=True, inplace=True)
        
        # Switch points
        lag_relCueCol = sim_data.relCueCol.eq(sim_data.relCueCol.shift())
        switch_points = np.where(lag_relCueCol == False)[0]
        switch_points = np.append(switch_points, len(sim_data))
        
        for starti, endi in af.pairwise(switch_points):
            nover = endi - starti - (2 * bin_size)
            for vari in varList:
                # First 15 trials after switch
                post15_data = sim_data.loc[starti:
                                           (starti + bin_size - 1)][[validity, vari]]
                post15[vari].append(
                    np.nanmean(post15_data[post15_data[validity] == 0][vari]) -\
                    np.nanmean(post15_data[post15_data[validity] == 1][vari])
                    )
                # Trials 15 to 30
                middle15_data = sim_data.loc[(starti + bin_size):
                                             (starti + 2 * bin_size - 1)][[validity, vari]]
                middle15[vari].append(
                    np.nanmean(middle15_data[middle15_data[validity] == 0][vari]) -\
                    np.nanmean(middle15_data[middle15_data[validity] == 1][vari])
                    )
                # All trials except for first 30
                left_data = sim_data.loc[(starti + 2 * bin_size):
                                         (endi - 1)][[validity, vari]]
                leftover[vari].append(
                    np.nanmean(left_data[left_data[validity] == 0][vari]) -\
                    np.nanmean(left_data[left_data[validity] == 1][vari])
                    )
                # The 15 trials before switch
                pre15_data = sim_data.loc[(endi - bin_size):
                                          (endi - 1)][[validity, vari]]
                pre15[vari].append(
                    np.nanmean(pre15_data[pre15_data[validity] == 0][vari]) -\
                    np.nanmean(pre15_data[pre15_data[validity] == 1][vari])
                    )
                # Last 15 trials or less if there were less than 45 trials
                # between switches
                if nover >= bin_size:
                    last15_data = sim_data.loc[(endi - bin_size):
                                               (endi - 1)][[validity, vari]]
                else:
                    last15_data = sim_data.loc[(endi - nover):
                                               (endi - 1)][[validity, vari]]
                last15[vari].append(
                    np.nanmean(last15_data[last15_data[validity] == 0][vari]) -\
                    np.nanmean(last15_data[last15_data[validity] == 1][vari])
                    )

    return post15, middle15, leftover, last15, pre15



#------------------------------------------------------------------------- End
