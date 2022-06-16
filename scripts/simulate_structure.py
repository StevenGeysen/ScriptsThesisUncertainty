#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Simulate experiment -- Version 3
Last edit:  2022/06/16
Author(s):  Geysen, Steven (01611639; SG)
Notes:      - Simulate the task used by Marzecova et al. (2019)
            - Release notes:
                * New beginings
                * Simulations only
            
To do:      - Implement in other scripts
            
Comments:   SG: Simulation function returns pandas.DataFrame
            
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
    DataFrame.csv
        The colums contain:
            0. 'id' - participants id
            1. 'trial' - trial # of the task
            2. 'relCueCol' - colour of the relevant cue (0: white / 1: black)
            3. 'relCue' - direction of the relevant cue (0: left / 1: right)
            4. 'irrelCue'- direction of the irrelevant cue (0: left / 1: right)
            5. 'targetLoc' - location of the target (0: left / 1: right)
            6. 'validity' - trial validity
    """

    # DataFrame
    #----------
    column_list = [
        'id', 'trial',
        'relCueCol', 'relCue', 'irrelCue', 'targetLoc', 'validity'
        ]
    dataDict = {keyi: [] for keyi in column_list}
    dataDict['id'] = ppnr

    # Variables
    #----------
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



#------------------------------------------------------------------------- End
