#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Assisting functions -- Version 1.1
Last edit:  2022/07/05
Author(s):  Geysen, Steven (SG)
Notes:      - Assisting functions to reduce repetition in other functions
                * labelDict
                * policy
                * save_data
            - Release notes:
                * Policy ()
To do:      - 
            
Questions:  
Comments:   
Sources:    https://goodresearch.dev/setup.html
"""



#%% ~~ Imports ~~ %%#


import numpy as np
import pandas as pd

import src.behavioural_functions as bf
import src.sim_functions as sf



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
            'R': 'Random',
            'PP': 'Participant'}


def policy(asm, Qest, beta=None):
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
        SoftMax temperature. The default is None.

    Returns
    -------
    int
        Selected cue.
    """

    # Argmax
    if asm.upper() == 'ARG':
        selcue = np.argmax(Qest)
        probcue = 1
    
    # SoftMax
    elif asm.upper() == 'SOFT':
        # Need beta value if SoftMax is used
        assert not beta is None, 'Missing beta value'
        
        ## Probability of cue 0
        probcue = np.exp(beta * Qest[0]) / \
            np.sum(np.exp(np.multiply(beta, Qest)))
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



# ------------------------------------------------------------------------ End
