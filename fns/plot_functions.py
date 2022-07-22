#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Plot functions -- Version 2.1
Last edit:  2022/07/15
Author(s):  Geysen, Steven (SG)
Notes:      - Functions used to plot output
                * Heatmap
                * Sanity checks
                    - Selection plot
                    - RT distributions
                * Learning curve
                * Stay behaviour
            - Release notes:
                * Added label x-axis
            
To do:      - Add functions of other often used plots
            - Learning curve with participant data
            
Comments:   
            
Sources:     https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
            https://stackoverflow.com/a/55768955
"""



#%% ~~ Imports ~~ %%#


import matplotlib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import fns.assisting_functions as af

from scipy import stats



#%% ~~ Sanity checks ~~ %%#
###########################


# ~~ Selection plot ~~ #
def selplot(data, model, plotnr, thetas=None, pp=''):
    """
    Plot cue selection of model

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the structure of the experiment and model
        behaviour.
    model : string
        Name of the used model:
            RW - Rescorla-Wagner
            H - RW-PH hybrid
            W - Win-stay-lose-shift
            R - Random
    thetas : list, array, tuple
        Parameter values.
    plotnr : int
        Plot number.
    pp : int, optional
        Number of the participant or simulation. The default is ''.

    Returns
    -------
    Plot.
    """


    plt.figure(plotnr)
    # Set title
    models = af.labelDict()
    if not model.upper() in ['W', 'R']:
        if model.upper()[:2] == 'RW':
            model_par = '$\u03B1$'
        else:
            model_par = '$\u03B7$'
        title = f'Cue selection {pp} {models[model.upper()]}: \
            {model_par} = {round(thetas[0], 4)}; $\u03B2$ = {round(thetas[1], 4)}'
    # Cue estimates
        plt.plot(data[[f'Qest_0_{model.upper()}']], label = 'Cue 0')
        plt.plot(data[[f'Qest_1_{model.upper()}']], label = 'Cue 1')
    else:
        title = f'Cue selection {pp} {models[model.upper()]}'
    plt.title(title)
    
    # Selected cue
    plt.plot(data[[f'selCue_{model.upper()}']], label = 'Selected cue')
    # True cue
    plt.plot(data[['relCueCol']], label = 'True cue', linestyle = '-.')
    
    plt.xlabel('trials')
    plt.legend()

    plt.show()


# ~~ RT distribution plot ~~ #
def rt_dist(data, model, thetas, plotnr, pp=''):
    """
    Plot distributions of response times, simulated by the model

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the structure of the experiment and model
        behaviour.
    model : string
        Name of the used model:
            RW - Rescorla-Wagner
            H - RW-PH hybrid
    thetas : list, array, tuple
        Parameter values.
    plotnr : int
        Plot number.
    pp : int, optional
        Number of the participant or simulation. The default is ''.

    Returns
    -------
    Plot.
    """

    # Wilhelm and Renee do not simulate RTs.
    assert not model.upper() in ['W', 'R'], 'Model has no simulated RT'

    # Response times
    rt_valid = np.where(data['relCue'] == data['targetLoc'],
                        data[f'rt_{model.upper()}'], np.nan)
    rt_invalid = np.where(data['relCue'] != data['targetLoc'],
                        data[f'rt_{model.upper()}'], np.nan)
    # Set title
    models = af.labelDict()
    if model.upper()[:2] == 'RW':
        model_par = '$\u03B1$'
    else:
        model_par = '$\u03B7$'
    title = f'RT distribution {pp} {models[model.upper()]}: \
        {model_par} = {round(thetas[0], 4)}; $\u03B2$ = {round(thetas[1], 4)}'
    
    plt.figure(plotnr)
    fig, ((ax0, ax1)) = plt.subplots(nrows = 1, ncols = 2)
    
    ax0.hist(rt_valid, bins = 30)
    ax0.set_title('Valid RT (s)')
    
    fig.suptitle(title, fontsize=14)
    
    ax1.hist(rt_invalid, bins = 30)
    ax1.set_title('Invalid RT (s)')

    plt.show()



#%% ~~ Ten simple rules ~~ %%#


# ~~ Learning curve ~~ #
def learning_curve(dataList, datadir, plotnr, wsls=False):
    """
    Plot learning curve
    The learning curves depict, trial by trial, the proportion of choosing the
    predictive cue (relCueCol).

    Parameters
    ----------
    dataList : list
        List containing the filenames of the data.
    datadir : Path
        Location of the data
    plotnr : int
        Plot number.
    wsls : bool, optional
        Set to True for plotting WSLS model. The default is False.

    Returns
    -------
    Plot.
    """

    # Preparation
    # -----------
    ##SG: 4 models, 640 trials
    LCmatrix = np.full((4, 640, len(dataList)), np.nan)
    
    for filei, file in enumerate(dataList):
        data = pd.read_csv(datadir / file, index_col='Unnamed: 0')
        
        DaphneList = []
        HugoList = []
        WilhelmList = []
        ReneeList = []
        
        # For each trial, is the predicitive cue selected,
        for triali, trial in data.iterrows():
            correct_Daphne = trial.relCueCol == trial.selCue_RW
            DaphneList.append(correct_Daphne > 0.5)
            
            correct_Hugo = trial.relCueCol == trial.selCue_H
            HugoList.append(correct_Hugo > 0.5)
                
            correct_Wilhelm = trial.relCueCol == trial.selCue_W
            WilhelmList.append(correct_Wilhelm > 0.5)
            
            correct_Renee = trial.relCueCol == trial.selCue_R
            ReneeList.append(correct_Renee > 0.5)
            
        LCmatrix[0, :, filei] = DaphneList
        LCmatrix[1, :, filei] = HugoList
        LCmatrix[2, :, filei] = WilhelmList
        LCmatrix[3, :, filei] = ReneeList

    # Plot
    # ----
    meanLC = np.nanmean(LCmatrix, axis=2)
    ## Switch points
    ##SG: Only once since all data used the same experimental structure (at
        # least for the simulations -- fix this  when using participant data).
    lag_relCueCol = data.relCueCol.eq(data.relCueCol.shift())
    switches = np.where(lag_relCueCol == False)[0][1:]
    
    plt.figure(plotnr)
    plt.xlabel('Trials')
    plt.ylabel('Probability of choosing correct')
    plt.ylim(-0.1, 1.1)
    
    # Plot all participants and not the mean
    plt.plot(meanLC[0, :], label = 'RW')
    plt.plot(meanLC[1, :], label = 'Hybrid')
    plt.plot(meanLC[3, :], label = 'Random')
    if wsls:
        ##SG: Makes the plot unclear.
        plt.plot(meanLC[2, :], label = 'WSLS')
    plt.vlines(switches, -0.05, 1.05, colors='black')
    
    plt.legend()
    plt.tight_layout()
    plt.suptitle(f'Mean learning curve over {len(dataList)} simulations', y=.99)

    plt.show()


# ~~ P(stay) ~~ #
def p_stay(dataList, datadir, plotnr):
    """
    Plot stay behaviour
    "P(stay) as a function of the reward on the last trial
    for each of the models with a particular set of parameters."
    -- Wilson and Collins (2019; box 2)

    Parameters
    ----------
    dataList : list
        List containing the filenames of the data.
    datadir : Path
        Location of the data
    plotnr : int
        Plot number.

    Returns
    -------
    Plot.
    """

    # Preparation
    # -----------
    # Models
    modelist = ['RW', 'H', 'W', 'R']
    # Line styles for plot
    stylist = ['--','-', '-.', ':']
    legendList = ['RW', 'Hybrid', 'WSLS', 'Random']
    
    noRewardList = ['RW_0', 'H_0', 'W_0', 'R_0']
    noRewardDict = {keyi:[] for keyi in noRewardList}
    RewardDict = {f'{keyi[:-1]}1':[] for keyi in noRewardList}
    
    for filei in dataList:
        data = pd.read_csv(datadir / filei, index_col='Unnamed: 0')
        for triali, trial in data.iterrows():
            if triali > 0:
                for modeli in modelist:
                    currSelection = trial[f'selCue_{modeli}']
                    prevSelection = data.loc[triali - 1, f'selCue_{modeli}']
                    # The same cue (1) or the other cue (0) is selected
                    compSelection = currSelection == prevSelection
                    
                    ## If the reward on the previous trial was 0
                    prevReward = data.loc[triali - 1, f'reward_{modeli}']
                    if prevReward == 0:
                        noRewardDict[f'{modeli}_0'].append(compSelection)
                    ## If the reward on the previous trial was 1
                    elif prevReward == 1:
                        RewardDict[f'{modeli}_1'].append(compSelection)
    # Mean and standard error
    relNRdict = {}
    for keyi in noRewardDict:
        relNRdict[keyi] = [np.nanmean(noRewardDict[keyi]),
                           stats.sem(noRewardDict[keyi])]
    relRdict = {}
    for keyi in RewardDict:
        relRdict[keyi] = [np.nanmean(RewardDict[keyi]),
                          stats.sem(RewardDict[keyi])]

    # Plot
    # ----
    plt.figure(plotnr)
    for stylei, modeli, labeli in zip(stylist, modelist, legendList):
        agentMatrix = np.asarray([[relNRdict[f'{modeli}_0'][0],
                                   relRdict[f'{modeli}_1'][0]],
                                  [relNRdict[f'{modeli}_0'][1],
                                   relRdict[f'{modeli}_1'][1]]])
        plt.errorbar(x=[0,1], y=agentMatrix[0, :],
                     yerr=agentMatrix[1, :],
                     label=labeli,
                     linestyle=stylei)
    ## Settings x-axis
    plt.xlabel('Reward on previous trial')
    plt.xticks((0, 1))
    ## Settings y-axis
    plt.ylabel('p(stay)')
    
    plt.title('Stay behaviour')
    plt.legend(loc='lower right')

    plt.show()



#%% ~~ Heatmap ~~ %%#
#####################


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={},
            cbarlabel='', row_name='', col_name='', **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data : np.darray
        A 2D numpy array of shape (M, N).
    row_labels : list, array
        A list or array of length M with the labels for the rows.
    col_labels : list, array
        A list or array of length N with the labels for the columns.
    ax : optional
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted. If
        not provided, use current axes or create a new one.
    cbar_kw : dict, optional
        A dictionary with arguments to `matplotlib.Figure.colorbar`.
    cbarlabel : str, optional
        The label for the colorbar.
    row_name : str, optional
        The name for the y-axis.
    col_name : str, optional
        The name for the x-axis.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()
        
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    
    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_xlabel(col_name)
    
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(row_labels)
    ax.set_ylabel(row_name)
    
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
             rotation_mode="anchor")
    
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
        
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.
        
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    
    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
        
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



# ------------------------------------------------------------------------ End
