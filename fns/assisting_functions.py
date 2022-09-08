#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Assisting functions -- Version 3.3
Last edit:  2022/09/08
Author(s):  Geysen, Steven (SG)
Notes:      - Assisting functions to reduce repetition
                * labelDict
                * policy
                * save_data
                * pairwise
            - Release notes:
                * Moved bin_switch to bf
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



# ------------------------------------------------------------------------ End
