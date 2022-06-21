#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Thesis Data Simulation -- Version 3
Last edit:  2022/06/16
Author(s):  Geysen, Steven (01611639; SG)
Notes:      - Based on the scripts in Ch6 of Modeling of Cognitive
                Processes (2020) and the data from Marzecova et al. (2019)
            - Release notes:
                * New beginings
                * Simulations only
                    Simulation function returns pandas.DataFrame
                * RT sampling
                    - No restrictions on RT
                    - When pe[selCue] = 0, RT = NaN
                    - When RT is infinite, RT = 1.7
                * Option to save simulated data
                    - Data is saved when file name is filled in
                    - Creates folder 'Simulations' when it not already exists
                * Alpha and beta loop
                
Comments:   SG: Change directory in settings.
            SG: Check RT in Simulation_3 for all previous attempts.
            
Sources:    https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00612/full
            https://docs.sympy.org/latest/modules/stats.html
            https://github.com/Kingsford-Group/ribodisomepipeline/blob/master/scripts/exGaussian.py
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html
            https://hannekedenouden.ruhosting.nl/RLtutorial/html/SimulationTopPage.html
            https://lindeloev.shinyapps.io/shiny-rt/
"""



#%% ~~ Imports ~~ %%#


import inspect
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats



#%% ~~ Settings ~~ %%#


# Number of simulations
N_SIMS = 5
# Start parameters
x0 = np.array([0.1, 1])
## Alpha options
alphaOptions = np.array([0.05, 0.1, 0.25, 0.5])
## Beta options
betaOptions = np.array([1, 1.5, 2, 2.5, 3, 15])
# Simulated model
sim_model = 'Hybrid'
# Set plot number
plotnr = 0
#%% Data storage
filename = inspect.getframeinfo(inspect.currentframe()).filename
cwd = os.path.dirname(os.path.abspath(filename))
SaveDataDir = cwd + '\Simulations'
if not os.path.isdir(SaveDataDir):
    os.mkdir(SaveDataDir)
os.chdir(SaveDataDir)

# SIM_DIR = Path.cwd() / 'Simulations'
# if not Path.exists(SIM_DIR):
#     Path.mkdir(SIM_DIR)


#%% ~~ Function ~~ %%#
######################


def sim_experiment(params=(0.1, 20), w0=0.5, ntrials=640, nstim=2, nswitch=7,
                   ppnr=1, model='RW', file_name=''):
    """
    Simulate data for/with the learning model.
    Simulates RT with an ex-Gaussian distribution as a function of PE.

    Parameters
    ----------
    params : tuple, list, array, optional
        Model Parameters
            * First parameter is the learning rate in the model.
            * Second parameter is the constant in the action selection method.
            * Last parameter is the starting value of alpha in hybrid model.
        The default is (0.1, 20).
    w0 : flt, optional
        Starting value of cue estimates.
        The default is 0.5.
    ntrials : int, optional
        Number of trials.
        The default is 640.
    nstim : int, optional
        Number of stimuli.
        The default is 2.
    nswitch : int, optional
        Number of switches in cue validity during the experiment.
        The default is 7.
    model : string, optional
        Learning model
            * 'rw' for Rescorla-Wagner
            * 'hybrid' for RW-PH hybrid model
        The default is 'RW'.
    file_name : string, optional
        Name af the csv file with the simulated data.
        If no name is filled in, the simulated data is stored in a cvs file.
        When filling in, no need to add '.csv' in the name.
        The default is ''.

    Returns
    -------
    DataFrame.csv
        The colums contain:
            0. 'id' - participants id
            1. 'trial' - trial # of the task
            2. 'relCue' - direction of the relevant cue (0: left / 1: right)
            3. 'irrelCue'- direction of the irrelevant cue (0: left / 1: right)
            4. 'targetLoc' - location of the target (0: left / 1: right)
            5. 'relCueCol' - colour of the relevant cue (0: white / 1: black)
            6. 'Choice' - selected cue
            7. 'Reward' - validity of the selected cue
            8. 'Cue_1 est' - estimated value of left cue
            9. 'Cue_2 est' - estimated value of right cue
            10. 'Cue_1 pe' - prediction error reflecting divergence from the prediction on current trial for left cue
            11. 'Cue_2 pe' - prediction error reflecting divergence from the prediction on current trial for right cue
            12. 'RT' - response time in s
            13. 'selPE' -- prediction error of selected cue
            14. 'selQ_est' -- estimated cue value of selected cue
            15. 'sumQ_est' -- sum of estimated cue values
            16. 'PE valid' -- prediction error in valid trials
            17. 'PE invalid' -- prediction error in invalid trials
            18. 'RT valid' -- reaction time in valid trials
            19. 'RT invalid' -- reaction time in invalid trials
            20. 'Validity' -- trial validity

    """


    # DataFrame
    #----------
    column_list = [
        'id', 'trial', 'relCue', 'irrelCue', 'relCueCol', 'targetLoc',
        'Choice', 'Reward', 'Cue_1 est', 'Cue_2 est', 'Cue_1 pe',
        'Cue_2 pe', 'RT'
        ]
    dataDict = {keyi: [] for keyi in column_list}
    dataDict['id'] = ppnr
    data = pd.DataFrame(columns=column_list)


    # Simulate data
    #--------------

    # Variables
    #----------
    # Task
    ## Colour of relevant cue
    relCueCol = np.random.randint(nstim)
    ## Probabilities of reward for each stim
    prob = w0 * np.ones(nstim)
    ## SG: For participants with an even number is the initial probability
        ## of the relevant cue 0.7
    if ppnr % 2 == 0:
        prob[relCueCol] = 0.7
    else:
        prob[relCueCol] = 0.85
    ## Trials where probability is switched
    switch = np.cumsum(np.random.randint(40, 120, size = nswitch))
    
    # Models
    ## Rewards
    reward = np.zeros(nstim)
    ## Prediction errors
    pe = np.zeros(nstim)
    ## Estimated cue value
    Q_est = np.full((ntrials + 1, nstim), w0)
    # Alpha
    if len(params) > 2:
        salpha = params[-1]
    else:
        salpha = 0.01
    alpha = np.full((ntrials + 1, nstim), salpha)
    
    # SoftMax
    ### Probabilities of cues
    pc = np.zeros((nstim))


    # Trial loop
    #-----------
    for triali in range(ntrials):
        dataDict['trial'] = triali
        # Switch probabilities
        if switch.shape[0] != 0:
            if triali == switch[0]:
                relCueCol = 1 - relCueCol
                
                if ppnr % 2 == 0:
                    ## For participants with an even number is in the
                    ## second half of trials the probability
                    ## of the relevant cue 0.85
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
                
        # Stimuli
        stim = np.random.choice(nstim, nstim, p =  [0.5, 0.5])
        
        # Action selection methods
        #-------------------------
        ## Random for first trial
        if triali == 0:
            selCue = np.random.randint(nstim)
        ## SoftMax in following trials
        else:
            ## Probabilities of the cues
            pc[relCueCol] = np.exp(params[1] * Q_est[triali, relCueCol]) / np.sum(np.exp(params[1] * Q_est[triali, :]))
            pc[1 - relCueCol] = np.exp(params[1] * Q_est[triali, 1 - relCueCol]) / np.sum(np.exp(params[1] * Q_est[triali, :]))
            ## SG: If prob of cue 0 is smaller than a random value
                ## between 0 and 1, follow cue 1.
            temp = np.random.rand() <= pc[relCueCol]
            selCue = int(temp == relCueCol)
            
        ## SG: Target is 'randomly' selected between 0 and 1 with the
            ## probability of the relevant cue. relCueCol- is to not always
            ## have 0 with the highest probability. abs() to prevent the
            ## target from being -1.
        target = abs(relCueCol - np.random.choice(nstim, p = [prob[stim[relCueCol]], 1 - prob[stim[relCueCol]]]))
        # Reward calculation
        ## Based on validity
        ## AM: If cue==target, reward = 1; if cue!=target reward = 0
        reward[selCue] = stim[selCue] == target
        reward[1 - selCue] = stim[1 - selCue] == target
        
        # Update rules
        #-------------
        pe[selCue] = reward[selCue] - Q_est[triali, selCue]
        # Rescorla-Wagner
        if model.upper() == 'RW':
            Q_est[triali + 1, selCue] = Q_est[triali, selCue] + params[0] * pe[selCue]
        
        # RW-PH hybrid
        elif model.upper() == 'HYBRID':
            alpha[triali + 1, selCue] = params[0] * np.abs(pe[selCue]) + (1 - params[0] * alpha[triali, selCue])
            Q_est[triali + 1, selCue] = Q_est[triali, selCue] + alpha[triali, selCue] * pe[selCue]
        
        Q_est[triali + 1, 1 - selCue] = Q_est[triali, 1 - selCue]
        
        # Reaction time
        try:
            ## stats.exponnorm.rvs(K = tau / sigma, loc = mu, scale = sigma)
            ## SG: Sigma and mu from exgaus fit original data.
            RT = stats.exponnorm.rvs(K = abs(pe[selCue]) / 0.02635, loc = 0.3009, scale = 0.02635)
        except:
            RT = np.nan
            print(f'Without cutoff {pe[selCue]}')
            print(f'trial {triali}')
            print(reward[stim[selCue]])
            print(Q_est[stim[selCue]])
        
        if RT == float('inf'):
            RT = 1.7
    
        # DataFrame
        #----------
        ## 'id', 'trial', 'relCue', 'irrelCue', 'relCueCol', 'targetLoc', 'choice', 'Reward', 'Cue_1 est', 'Cue_2 est', 'Cue_1 pe', 'Cue_2 pe', 'RT'
        data.loc[triali] = [ppnr, triali, stim[relCueCol], stim[1 - relCueCol], relCueCol, target, selCue, reward[selCue], Q_est[triali, 0], Q_est[triali, 1] , pe[0], pe[1], RT]
        
    ## Prediction error of selected cue
    data['selPE'] = np.where(data['Choice'] == 0, data['Cue_1 pe'], data['Cue_2 pe'])
    ## Estimated cue value of selected cue
    data['selQ_est'] = np.where(data['Choice'] == 0, data['Cue_1 est'], data['Cue_2 est'])
    ## Sum of estimated cue values
    data['sumQ_est'] = data['Cue_1 est'] + data['Cue_2 est']
    ## PE in valid trials
    data['PE valid'] = np.where(data['relCue'] == data['targetLoc'], data['selPE'], np.nan)
    ## PE in invalid trials
    data['PE invalid'] = np.where(data['relCue'] != data['targetLoc'], data['selPE'], np.nan)
    ## RT in valid trials
    data['RT valid'] = np.where(data['relCue'] == data['targetLoc'], data['RT'], np.nan)
    ## RT in invalid trials
    data['RT invalid'] = np.where(data['relCue'] != data['targetLoc'], data['RT'], np.nan)
    ## Validity
    data['Validity'] = np.where(data['relCue'] == data['targetLoc'], 1, 0)
    
    # Output dataframe
    ## Save to a cvs file if a file name is filled in
    if len(file_name) != False:
        data.to_csv(model + '_' + file_name + '.csv')
    return data



#%% ~~ Simulations ~~ %%#
#########################


## SG: ([x, y, z])
woc_pe_rt = np.full((N_SIMS * len(alphaOptions), 2, len(betaOptions)), np.nan)
woc_abspe_rt = np.full((N_SIMS * len(alphaOptions), 2, len(betaOptions)), np.nan)
woc_q_rt = np.full((N_SIMS * len(alphaOptions), 2, len(betaOptions)), np.nan)
woc_sum_rt = np.full((N_SIMS * len(alphaOptions), 2, len(betaOptions)), np.nan)
wocmeanpe = np.full((N_SIMS * len(alphaOptions), len(betaOptions)), np.nan)
wocmeanrt = np.full((N_SIMS * len(alphaOptions), 3, len(betaOptions)), np.nan)

woc_valpe_rt = np.full((N_SIMS * len(alphaOptions), 2, len(betaOptions)), np.nan)
woc_invalpe_rt = np.full((N_SIMS * len(alphaOptions), 2, len(betaOptions)), np.nan)


start_total = time.time()
# Alpha loop
for alphai, alpha in enumerate(alphaOptions):
    x0[0] = alpha
    # Beta loop
    for betai, beta in enumerate(betaOptions):
        x0[1] = beta
        start_sim = time.time()
        # Simulation loop
        for pp in range(N_SIMS):
            print('-----')
            print(pp, beta, alpha)
            
            # New simulated dataframe
            newSim = sim_experiment(params = x0, ppnr = pp, model = sim_model,
                                    file_name = f'alpha_{alpha}_beta_{beta}_nsim{pp}')
            wocmeanpe[pp + (alphai * N_SIMS), betai] = np.nanmean(newSim['selPE'])
            wocmeanrt[pp + (alphai * N_SIMS), 0, betai] = np.nanmean(newSim['RT'])
            wocmeanrt[pp + (alphai * N_SIMS), 1, betai] = np.nanmean(newSim['RT valid'])
            wocmeanrt[pp + (alphai * N_SIMS), 2, betai] = np.nanmean(newSim['RT invalid'])
            print(f"PE {newSim['selPE'].describe()}")
            
            # # Sanity checks
            # ## Selection plot
            # plt.figure(plotnr)
            # plt.title(f'Cue selection iteration {pp} / {x0[0]} / {x0[1]}')
            #     ## Cue estimates
            # plt.plot(newSim[['Cue_1 pe']], label = 'Cue 1')
            # plt.plot(newSim[['Cue_2 pe']], label = 'Cue 2')
            #     ## Selected cue
            # plt.plot(newSim[['Choice']], label = 'Selected cue')
            #     ## True cue
            # plt.plot(newSim[['relCueCol']], label = 'True cue', linestyle = '-.')
            #     ## Legend
            # plt.legend()
            
            # plt.show()
            # plotnr += 1
            
            # ## Validity effect
            # print(f" max RT {np.max(newSim['RT'])}")
            # print(f" min RT {np.min(newSim['RT'])}")
            # ### Boxplots
            # plt.figure(plotnr)
            # newSim.boxplot(column = ['RT valid', 'RT invalid'])
            # plt.show()
            # plotnr += 1
            # plt.figure(plotnr)
            # newSim.boxplot(column = ['PE valid', 'PE invalid'])
            
            # plt.show()
            # plotnr += 1
            
            # ## RT distribution plot
            # plt.figure(plotnr)
            # plt.hist(newSim[['RT valid']], bins = 30, alpha = 0.5, label='Valid')
            # plt.hist(newSim[['RT invalid']], bins = 30, alpha = 0.5, label='Invalid')
            # plt.legend()
            
            # plt.show()
            # plotnr += 1
            
            
            # plt.figure(plotnr)
            # fig, ((ax0, ax1)) = plt.subplots(nrows = 1, ncols = 2)
            # ax0.hist(newSim[['RT valid']], bins = 30)
            # ax0.set_title('RT valid')
            # fig.suptitle(f'RT distribution pp {pp + 1}, simulation {pp + 1} ({x0})', fontsize=14)
            # ax1.hist(newSim[['RT invalid']], bins = 30)
            # ax1.set_title('RT invalid')
            
            # plt.show()
            # plotnr += 1
            
            
            # Calculations
            ## Correlation
            woc_pe_rt[pp + (alphai * N_SIMS), :, betai] = stats.spearmanr(newSim['RT'], newSim['selPE'], nan_policy = 'omit')
            woc_abspe_rt[pp + (alphai * N_SIMS), :, betai] = stats.spearmanr(newSim['RT'], abs(newSim['selPE']), nan_policy = 'omit')
            # print(f"Correlation selPE-RT without cutoff {stats.spearmanr(newSim['RT'], newSim['selPE'], nan_policy = 'omit')[0]}")
            
            woc_q_rt[pp + (alphai * N_SIMS), :, betai] = stats.spearmanr(newSim['RT'], newSim['selQ_est'], nan_policy = 'omit')
            # print(f"Correlation selQ_est-RT without cutoff {stats.spearmanr(newSim['RT'], newSim['selQ_est'], nan_policy = 'omit')[0]}")
            
            woc_sum_rt[pp + (alphai * N_SIMS), :, betai] = stats.spearmanr(newSim['RT'], newSim['sumQ_est'], nan_policy = 'omit')
            # print(f"Correlation sumQ_est-RT without cutoff {stats.spearmanr(newSim['RT'], newSim['sumQ_est'], nan_policy = 'omit')[0]}")
            
            woc_valpe_rt[pp + (alphai * N_SIMS), :, betai] = stats.spearmanr(newSim['RT valid'], newSim['PE valid'], nan_policy = 'omit')
            woc_invalpe_rt[pp + (alphai * N_SIMS), :, betai] = stats.spearmanr(newSim['RT invalid'], newSim['PE invalid'], nan_policy = 'omit')
            
        
        end_sim = time.time()
        print(f'duration of alpha {alphai + 1} is {round(end_sim - start_sim, 2)} seconds')

end_total = time.time()
print(f'total duration: {round((end_total - start_total) / 60, 2)} minutes')



#------------------------------------------------------------------------- End
