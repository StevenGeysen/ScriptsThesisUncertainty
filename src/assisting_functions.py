#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Assisting functions -- Version 1
Last edit:  2022/07/04
Author(s):  Geysen, Steven (SG)
Notes:      - Assisting functions to reduce repetition in other functions
                * labelDict
                * pp_models
                * sim_models
                * save_data
            - Release notes:
                * Initial commit
To do:      - 
            
Questions:  
Comments:   
Sources:    
"""



#%% ~~ Imports ~~ %%#



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


def pp_models():
    """
    Participant dictionary
    Contains the functions of the different behavioural models.
    """

    return {'RW': bf.ppRW_1c,
            'RW2': bf.ppRW_2c,
            'H': bf.ppHybrid_1c,
            'H2': bf.ppHybrid_2c,
            'W': bf.ppWSLS,
            'R': bf.ppRandom}


def sim_models():
    """
    Simulation dictionary
    Contains the functions of the different simulation models.
    """

    return {'RW': sf.simRW_1c,
            'RW2': sf.simRW_2c,
            'H': sf.simHybrid_1c,
            'H2': sf.simHybrid_2c,
            'W': sf.simWSLS,
            'R': sf.simRandom}


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
