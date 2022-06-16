#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Thesis Data models simulations-- Version 1
Last edit:  2021/08/13
Author(s):  Geysen, Steven (01611639; SG)
Notes:      - Based on Verguts & Calderon
            - Data from Marzecova et al. (2019)
            - 4 spaces for tabs
            - Release notes
                * Adapted DataModel 3.0.1 to work on simulated data
                * Only fast models
                    - f[model name]_1c for 1 cue update
                    - f[model name]_2c for 2 cue update
                * PE in output
            
Comments:   SG: Check directory
            SG: Data in numpy without titles
"""



#%% ~~ Imports ~~ %%#


import os
import numpy as np



#%% ~~ Load data ~~ %%#


# ### Opening the data files
cwd = os.getcwd()
os.chdir(cwd + '\Simulations')
SimData = np.zeros((2,6))

# The data files were made with Simulations.
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


#%% Original data

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



#%% ~~ Functions ~~ %%#
#######################


#%% ~~ Models (1 cue) ~~ %%#
#--------------------------#


# ~~ Rescorla - Wagner ~~ #
def frw_1c(parameters=(0.01, 0.5), relCue=2, irrelCue=3, relCueCol=4, targetLoc=5, data=SimData):
    """
    Rescorla - Wagner learning model
    Action selection methods calculations in model.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is the learning rate in the model.
        Second parameter is constant in action selection method.
        The default is (0.01, 0.5).
    relCue : int
        Direction of the relevant cue (<=1).
    irrelCue : int
        Direction of the irrelevant cue (<=1).
    relCueCol : int
        Number for color of the relevant cue.
    targetLoc : int
        location of the target (<=1).
    data : np.ndarray
        DESCRIPTION.

    Returns
    -------
    Q_est : np.ndarray
        Estimated values.

    """


    # Variables
    #----------
    
    # Model
    ## Number of trials
    n_trials = data.shape[0]
    ## Number of cues
    n_cues = int(np.amax(data[:, relCueCol])) + 1
    ## Rewards
    reward = np.zeros((n_trials, n_cues + 1))
    ## Prediction error
    pe = np.zeros(n_trials, )
    ## Estimated value of cue
    Q_est = np.full((n_trials, n_cues), 1/n_cues)
    
    # Action selection
    ## Selected cues
    selcues = np.zeros((n_trials), dtype = int)
    ## SoftMax
    ## Probabilities of cues
    pc = np.zeros((n_trials, n_cues))
    ##
    temp = np.zeros(n_trials, )


    # Trial loop
    for trial in range(n_trials - 1):
        # Select cue
        ## Random for first trial
        if trial == 0:
            selcues[trial] = np.random.randint(2)
        else:
            ## Probabilities of the cues
            pc[trial, int(data[trial, relCueCol])] = np.exp(parameters[1] * Q_est[trial, int(data[trial, relCueCol])]) / np.sum(np.exp(parameters[1] * Q_est[trial, :]))
            pc[trial, int(1 - (data[trial, relCueCol]))] = np.exp(parameters[1] *  Q_est[trial, int(1 - (data[trial, relCueCol]))]) /  np.sum(np.exp(parameters[1] * Q_est[trial, :]))
            
            ## Random value for comparison
            y = np.random.rand()
            ## If prob of cue 0 is smaller than y, follow cue 1
            # temp[trial] = y >= pc[trial, int(data[trial, relCueCol] - 1)]
            temp[trial] = y <= pc[trial, int(data[trial, relCueCol])]
            ## Action selection
            selcues[trial] = temp[trial] == int(data[trial, relCueCol])
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward =1, if cue!=target reward = 0
        reward[trial, int(data[trial, relCueCol])] = data[trial, relCue] == data[trial, targetLoc]
        reward[trial, int(1 - (data[trial, relCueCol]))] = data[trial, irrelCue] == data[trial, targetLoc]
        
        # Update rule (RW)
        #-----------------
        pe[trial] = reward[trial, selcues[trial]] - Q_est[trial, selcues[trial]]
        Q_est[trial + 1, selcues[trial]] = Q_est[trial, selcues[trial]] + parameters[0] * pe[trial]
        Q_est[trial + 1, 1 - selcues[trial]] = Q_est[trial, 1 - selcues[trial]]

    return Q_est, selcues, pe


# ~~ RW-PH Hybrid ~~ #
def fhybrid_1c(parameters=(0.01, 0.5), relCue=2, irrelCue=3, relCueCol=4, targetLoc=5, data=SimData, salpha=0.01):
    """
    RW-PH hybrid model
    Action selection methods calculations in model.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is eta.
        Second parameter is constant in action selection method.
        The default is (0.01, 0.5).
    relCue : int
        Direction of the relevant cue (<=1).
    irrelCue : int
        Direction of the irrelevant cue (<=1).
    relCueCol : int
        Number for color of the relevant cue.
    targetLoc : int
        Location of the target (<=1).
        Controls level of influence from past trials to the currenct learning rate.
    data : array
        DESCRIPTION.
    salpha : TYPE, optional
        Start of alpha. The default is 0.01.

    Returns
    -------
    Q_est : TYPE
        Estimated values.

    """


    # Variables
    ## Number of trials
    n_trials = data.shape[0]
    ## Number of cues
    n_cues = int(np.amax(data[:, relCueCol])) + 1
    # Alpha
    alpha = np.full((n_trials, n_cues), salpha)
    ## Rewards
    reward = np.zeros((n_trials, n_cues + 1))
    ## Prediction error
    pe = np.zeros(n_trials, )
    ## Estimated value of cue
    Q_est = np.full((n_trials, n_cues), 1/n_cues)
    
    # Action selection
    ## Selected cues
    selcues = np.zeros((n_trials), dtype = int)
    ## SoftMax
    ## Probabilities of cues
    pc = np.zeros((n_trials, n_cues))
    ##
    temp = np.zeros(n_trials, )


    # Trial loop
    for trial in range(n_trials - 1):
        # Select cue
        ## Random for first trial
        if trial == 0:
            selcues[trial] = np.random.randint(2)
        else:
            ## Probabilities of the cues
            # pc[trial, int(data[trial, relCueCol] - 1)] = np.exp(parameters[1] * Q_est[trial + 1, int(data[trial, relCueCol] - 1)]) / np.sum(np.exp(parameters[1] * Q_est[trial, int(data[trial, relCueCol] - 1)]))
            pc[trial, int(data[trial, relCueCol])] = np.exp(parameters[1] * Q_est[trial, int(data[trial, relCueCol])]) / np.sum(np.exp(parameters[1] * Q_est[trial, :]))
            # pc[trial, int(1 - (data[trial, relCueCol] - 1))] = np.exp(parameters[1] *  Q_est[trial + 1, int(1 - (data[trial, relCueCol] - 1))]) / np.sum(np.exp(parameters[1] * Q_est[trial, int(1 - (data[trial, relCueCol] - 1))]))
            pc[trial, int(1 - (data[trial, relCueCol]))] = np.exp(parameters[1] *  Q_est[trial, int(1 - (data[trial, relCueCol]))]) /  np.sum(np.exp(parameters[1] * Q_est[trial, :]))
            
            ## Random value for comparison
            y = np.random.rand()
            ## If prob of cue 0 is smaller than y, follow cue 1
            # temp[trial] = y >= pc[trial, int(data[trial, relCueCol] - 1)]
            temp[trial] = y <= pc[trial, int(data[trial, relCueCol])]
            ## Action selection
            selcues[trial] = temp[trial] == int(data[trial, relCueCol])
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward =1, if cue!=target reward = 0
        reward[trial, int(data[trial, relCueCol])] = data[trial, relCue] == data[trial, targetLoc]
        reward[trial, int(1 - (data[trial, relCueCol]))] = data[trial, irrelCue] == data[trial, targetLoc]
        
        # Hybrid
        #-------
        # Alpha (PH)
        alpha[trial + 1, selcues[trial]] = parameters[0] * np.abs(reward[trial, selcues[trial]] - Q_est[trial, selcues[trial]]) + (1 - parameters[0]) * alpha[trial, selcues[trial]]
        # Update Q (RW)
        pe[trial] = reward[trial, selcues[trial]] - Q_est[trial, selcues[trial]]
        Q_est[trial + 1, selcues[trial]] = Q_est[trial, selcues[trial]] + alpha[trial, selcues[trial]] * pe[trial]
        Q_est[trial + 1, 1 - selcues[trial]] = Q_est[trial, 1 - selcues[trial]]

    return Q_est, selcues, pe



#%% ~~ Models (2 cues) ~~ %%#
#---------------------------#


# ~~ Rescorla - Wagner ~~ #
def frw_2c(parameters=(0.01, 0.5), relCue=2, irrelCue=3, relCueCol=4, targetLoc=5, data=SimData):
    """
    Rescorla - Wagner learning model
    Action selection methods calculations in model.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is the learning rate in the model.
        Second parameter is constant in action selection method.
        The default is (0.01, 0.5).
    relCue : int
        Direction of the relevant cue (<=1).
    irrelCue : int
        Direction of the irrelevant cue (<=1).
    relCueCol : int
        Number for color of the relevant cue.
    targetLoc : int
        location of the target (<=1).
    data : np.ndarray
        DESCRIPTION.

    Returns
    -------
    Q_est : np.ndarray
        Estimated values.

    """


    # Variables
    #----------
    
    # Model
    ## Number of trials
    n_trials = data.shape[0]
    ## Number of cues
    n_cues = int(np.amax(data[:, relCueCol])) + 1
    ## Rewards
    reward = np.zeros((n_trials, n_cues + 1))
    ## Prediction error
    pe = np.zeros((n_trials, n_cues))
    ## Estimated value of cue
    Q_est = np.full((n_trials, n_cues), 1/n_cues)
    
    # Action selection
    ## Selected cues
    selcues = np.zeros((n_trials), dtype = int)
    ## SoftMax
    ## Probabilities of cues
    pc = np.zeros((n_trials, n_cues))
    ##
    temp = np.zeros(n_trials, )


    # Trial loop
    for trial in range(n_trials - 1):
        # Select cue
        ## Random for first trial
        if trial == 0:
            selcues[trial] = np.random.randint(2)
        else:
            ## Probabilities of the cues
            pc[trial, int(data[trial, relCueCol])] = np.exp(parameters[1] * Q_est[trial, int(data[trial, relCueCol])]) / np.sum(np.exp(parameters[1] * Q_est[trial, :]))
            pc[trial, int(1 - (data[trial, relCueCol]))] = np.exp(parameters[1] *  Q_est[trial, int(1 - (data[trial, relCueCol]))]) /  np.sum(np.exp(parameters[1] * Q_est[trial, :]))
            
            ## Random value for comparison
            y = np.random.rand()
            ## If prob of cue 0 is smaller than y, follow cue 1
            # temp[trial] = y >= pc[trial, int(data[trial, relCueCol] - 1)]
            temp[trial] = y <= pc[trial, int(data[trial, relCueCol])]
            ## Action selection
            selcues[trial] = temp[trial] == int(data[trial, relCueCol])
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward =1, if cue!=target reward = 0
        reward[trial, int(data[trial, relCueCol] - 1)] = data[trial, relCue] == data[trial, targetLoc]
        reward[trial, int(1 - (data[trial, relCueCol] - 1))] = data[trial, irrelCue] == data[trial, targetLoc]
        
        # Update rule (RW)
        #-----------------
        pe[trial, selcues[trial]] = reward[trial, selcues[trial]] - Q_est[trial, selcues[trial]]
        pe[trial, 1-selcues[trial]] = reward[trial, 1-selcues[trial]] - Q_est[trial, 1-selcues[trial]]
        Q_est[trial + 1, selcues[trial]] = Q_est[trial, selcues[trial]] + parameters[0] * pe[trial, selcues[trial]]
        Q_est[trial + 1, 1-selcues[trial]] = Q_est[trial, 1-selcues[trial]] + parameters[0] * pe[trial, 1-selcues[trial]]

    return Q_est, selcues, pe


# ~~ RW-PH Hybrid ~~ #
def fhybrid_2c(parameters=(0.01, 0.5), relCue=2, irrelCue=3, relCueCol=4, targetLoc=5, data=SimData, salpha=0.01):
    """
    RW-PH hybrid model
    Action selection methods calculations in model.

    Parameters
    ----------
    parameters : tuple, list, array
        First parameter is eta.
        Second parameter is constant in action selection method.
        The default is (0.01, 0.5).
    relCue : int
        Direction of the relevant cue (<=1).
    irrelCue : int
        Direction of the irrelevant cue (<=1).
    relCueCol : int
        Number for color of the relevant cue.
    targetLoc : int
        Location of the target (<=1).
        Controls level of influence from past trials to the currenct learning rate.
    data : array
        DESCRIPTION.
    salpha : TYPE, optional
        Start of alpha. The default is 0.01.

    Returns
    -------
    Q_est : TYPE
        Estimated values.

    """


    # Variables
    ## Number of trials
    n_trials = data.shape[0]
    ## Number of cues
    n_cues = int(np.amax(data[:, relCueCol])) + 1
    # Alpha
    alpha = np.full((n_trials, n_cues), salpha)
    ## Rewards
    reward = np.zeros((n_trials, n_cues + 1))
    ## Prediction error
    pe = np.zeros((n_trials, n_cues))
    ## Estimated value of cue
    Q_est = np.full((n_trials, n_cues), 1/n_cues)
    
    # Action selection
    ## Selected cues
    selcues = np.zeros((n_trials), dtype = int)
    ## SoftMax
    ## Probabilities of cues
    pc = np.zeros((n_trials, n_cues))
    ##
    temp = np.zeros(n_trials, )


    # Trial loop
    for trial in range(n_trials - 1):
        # Select cue
        ## Random for first trial
        if trial == 0:
            selcues[trial] = np.random.randint(2)
        else:
            ## Probabilities of the cues
            # pc[trial, int(data[trial, relCueCol] - 1)] = np.exp(parameters[1] * Q_est[trial + 1, int(data[trial, relCueCol] - 1)]) / np.sum(np.exp(parameters[1] * Q_est[trial, int(data[trial, relCueCol] - 1)]))
            pc[trial, int(data[trial, relCueCol])] = np.exp(parameters[1] * Q_est[trial, int(data[trial, relCueCol])]) / np.sum(np.exp(parameters[1] * Q_est[trial, :]))
            # pc[trial, int(1 - (data[trial, relCueCol] - 1))] = np.exp(parameters[1] *  Q_est[trial + 1, int(1 - (data[trial, relCueCol] - 1))]) / np.sum(np.exp(parameters[1] * Q_est[trial, int(1 - (data[trial, relCueCol] - 1))]))
            pc[trial, int(1 - (data[trial, relCueCol]))] = np.exp(parameters[1] *  Q_est[trial, int(1 - (data[trial, relCueCol]))]) /  np.sum(np.exp(parameters[1] * Q_est[trial, :]))
            
            ## Random value for comparison
            y = np.random.rand()
            ## If prob of cue 0 is smaller than y, follow cue 1
            # temp[trial] = y >= pc[trial, int(data[trial, relCueCol] - 1)]
            temp[trial] = y <= pc[trial, int(data[trial, relCueCol])]
            ## Action selection
            selcues[trial] = temp[trial] == int(data[trial, relCueCol])
        
        # Reward calculations
        ## Based on validity
        ##AM: If cue==target reward =1, if cue!=target reward = 0
        reward[trial, int(data[trial, relCueCol] - 1)] = data[trial, relCue] == data[trial, targetLoc]
        reward[trial, int(1 - (data[trial, relCueCol] - 1))] = data[trial, irrelCue] == data[trial, targetLoc]
        
        # Hybrid
        #-------
        # Alpha (PH)
        alpha[trial + 1, selcues[trial]] = parameters[0] * np.abs(reward[trial, selcues[trial]] - Q_est[trial, selcues[trial]]) + (1 - parameters[0]) * alpha[trial, selcues[trial]]
        # Update Q (RW)
        pe[trial + 1, selcues[trial]] = reward[trial, selcues[trial]] - Q_est[trial, selcues[trial]]
        pe[trial + 1, 1-selcues[trial]] = reward[trial, 1-selcues[trial]] - Q_est[trial, 1-selcues[trial]]
        Q_est[trial + 1, selcues[trial]] = Q_est[trial, selcues[trial]] + alpha[trial, selcues[trial]] * pe[trial, selcues[trial]]
        Q_est[trial + 1, 1-selcues[trial]] = Q_est[trial, 1-selcues[trial]] + alpha[trial, 1-selcues[trial]] * pe[trial, 1 - selcues[trial]]

    return Q_est, selcues, pe



#------------------------------------------------------------------------- End
