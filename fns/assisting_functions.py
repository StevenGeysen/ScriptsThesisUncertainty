#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Assisting functions -- Version 3.2.1
Last edit:  2022/09/03
Author(s):  Geysen, Steven (SG)
Notes:      - Assisting functions to reduce repetition
                * labelDict
                * policy
                * save_data
                * pairwise
                * bin_switch
            - Release notes:
                * Add bin_size to bin_switch
To do:      - 
            
Questions:  
Comments:   
Sources:    https://goodresearch.dev/setup.html
            https://docs.python.org/3/library/itertools.html#itertools.pairwise
            https://doi.org/10.1016/j.cub.2021.12.006
"""



#%% ~~ Imports ~~ %%#


import itertools

import numpy as np
import pandas as pd



#%% ~~ Support ~~ %%#
#####################


def labelDict():
    """
    Label dictionary
    Contains the full names of the short labels.
    """

    return {'RW': 'Rescorla-Wagner',
            'H': 'RW-PH Hybrid',
            'W': 'WSLS',
            'M': 'Meta learner',
            'R': 'Random',
            'PP': 'Participant'}


def policy(asm, Qest, beta=-1, bias=0):
    """
    Policy
    Select cue from with the argmax or SoftMax action selection method.

    Parameters
    ----------
    asm : string
        Action selection method.
    Qest : itarable
        Estimated values of previous trial.
    beta : float, optional
        SoftMax temperature (inverse temperature in biased SoftMax).
        The default is -1.
    bias : float, optional
        The bias term of the biased SoftMax. The default is 0.

    Returns
    -------
    int
        Selected cue.
    """

    asm = asm.upper()
    # Argmax
    if asm == 'ARG':
        selcue = np.argmax(Qest)
        probcue = 1
    
    # SoftMax
    elif asm == 'SOFT':
        ## Need beta value if SoftMax is used
        assert beta > 0, 'Missing beta value'
        
        ## Probability of cue 0
        probcue = 1 / (1 + np.exp(-beta  * (Qest[0] - Qest[1] + bias)))
        ##SG: If the probability of cue 0 is smaller than a random value,
            # follow cue 1.
        temp = np.random.rand() <= probcue
        ## Action selection
        selcue = int(temp == 0)

    return selcue, probcue


def save_data(dataDict, strucData, column_list):
    """
    Save model estimates
    Add the model's behaviour to the behavioural data or experiment structure.
    """

    # Save data
    modelData = pd.DataFrame(dataDict, columns=column_list)
    ## Correct indexes
    modelData.reset_index(drop=True, inplace=True)
    strucData.reset_index(drop=True, inplace=True)
    
    modelData = pd.concat([strucData, modelData], axis=1)

    # Output dataframe
    return modelData


def pairwise(iterable):
    """
    pairwise('ABCDEFG') --> AB BC CD DE EF FG
    https://docs.python.org/3/library/itertools.html#itertools.pairwise
    """
    a, b = itertools.tee(iterable)
    next(b, None)

    return zip(a, b)


def bin_switch(data, varList, bin_size=15):
    """
    Bin data before and after switch

    Parameters
    ----------
    data : pd.DataFrame
        Data of the experiment.
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

    # Variables
    #----------
    # Number of participants
    npp = data['id'].max()
    
    # Trial bins
    post15 = {vari:[] for vari in varList}
    middle15 = {vari:[] for vari in varList}
    leftover = {vari:[] for vari in varList}
    last15 = {vari:[] for vari in varList}
    pre15 = {vari:[] for vari in varList}

    for ppi in range(npp):
        ## Skip pp6 (not in data)
        if ppi + 1 == 6:
            continue
        ## Use only data from pp
        pp_data = data[data['id'] == ppi + 1]
        pp_data.reset_index(drop=True, inplace=True)
        
        # Switch points
        lag_relCueCol = pp_data.relCueCol.eq(pp_data.relCueCol.shift())
        switch_points = np.where(lag_relCueCol == False)[0]
        switch_points = np.append(switch_points, len(pp_data))
        
        for starti, endi in pairwise(switch_points):
            nover = endi - starti - (2 * bin_size)
            for vari in varList:
                post15[vari].append(
                    pp_data.loc[starti:(starti + bin_size - 1)][vari].to_numpy()
                    )
                middle15[vari].append(
                    pp_data.loc[(starti + bin_size):(starti + 2 * bin_size - 1)][vari].to_numpy()
                    )
                leftover[vari].append(
                    pp_data.loc[(starti + 2 * bin_size):(endi - 1)][vari].to_numpy()
                    )
                pre15[vari].append(
                    pp_data.loc[(endi - bin_size):(endi - 1)][vari].to_numpy()
                    )
                ##SG: Last 15 trials or less if there were less than 45 trials
                    # between switches.
                if nover >= bin_size:
                    last15[vari].append(
                        pp_data.loc[(endi - bin_size):(endi - 1)][vari].to_numpy()
                        )
                else:
                    last15[vari].append(
                        pp_data.loc[(endi - nover):(endi - 1)][vari].to_numpy()
                        )

    return post15, middle15, leftover, last15, pre15



# ------------------------------------------------------------------------ End
