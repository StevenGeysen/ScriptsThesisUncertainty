#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Plot functions -- Version 3.3
Last edit:  2022/09/13
Author(s):  Geysen, Steven (SG)
Notes:      - Functions used to plot output
                * Sanity checks
                    - Selection plot
                    - RT distributions
                    - PE validity effect
                * Learning curve
                * PE curve
                * Stay behaviour
                * Heatmaps
            - Release notes:
                * Corrected learning curve
            
To do:      - Add functions of other often used plots
            - Make plots work with participant data
            - Add new model
            
Comments:   
            
Sources:    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
            https://stackoverflow.com/a/55768955
            https://stackoverflow.com/a/45842334
            https://stackoverflow.com/a/8409110
"""



#%% ~~ Imports ~~ %%#


import matplotlib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import fns.assisting_functions as af

from scipy import signal, stats



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
            M - Meta learner
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

    models = af.labelDict()
    model = model.upper()

    plt.figure(plotnr)
    # Set title
    if not model in ['W', 'R']:
        if model[:2] == 'RW':
            model_par = '$\u03B1$'
        elif model == 'H':
            model_par = '$\u03B7$'
        title = f'Cue selection {pp} {models[model]}: \
            {model_par} = {round(thetas[0], 4)}; $\u03B2$ = {round(thetas[1], 4)}'
    # Cue estimates
        plt.plot(data[[f'Qest_0_{model}']], label='Cue 0')
        plt.plot(data[[f'Qest_1_{model}']], label='Cue 1')
    else:
        title = f'Cue selection {pp} {models[model]}'
    plt.title(title)
    
    # Selected cue
    plt.plot(data[[f'selCue_{model}']], label='Selected cue')
    # True cue
    plt.plot(data[['relCueCol']], label='True cue', linestyle='-.')
    
    plt.xlabel('trials')
    plt.legend()

    plt.show()


# ~~ RT distribution plot ~~ #
def rt_dist(data, model, thetas, pp=''):
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
            M - Meta learner
    thetas : list, array, tuple
        Parameter values.
    pp : int, optional
        Number of the participant or simulation. The default is ''.

    Returns
    -------
    Plot.
    """

    models = af.labelDict()
    model = model.upper()
    # Wilhelm and Renee do not simulate RTs.
    assert not model in ['W', 'R', 'M'], 'Model has no simulated RT'

    # Response times
    rt_valid = np.where(data['relCue'] == data['targetLoc'],
                        data[f'rt_{model}'], np.nan)
    rt_invalid = np.where(data['relCue'] != data['targetLoc'],
                        data[f'rt_{model}'], np.nan)
    plotdata = [rt_valid, rt_invalid]
    plabels = ['Valid RT (s)', 'Invalid RT (s)']
    
    # Set title
    if model[:2] == 'RW':
        model_par = '$\u03B1$'
    else:
        model_par = '$\u03B7$'
    title = f'RT distribution {pp} {models[model]}: \
        {model_par} = {round(thetas[0], 4)}; $\u03B2$ = {round(thetas[1], 4)}'
    
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(title, fontsize=14)
    
    for i, rtdata in enumerate(plotdata):
        axs[i].hist(rtdata, bins=30)
        axs[i].set_title(plabels[i])

    plt.show()


def pe_validity(model, dataList, datadir, wsls=False):
    """
    Box plot and violin plot
    Visual inspection of validity effect in perdiction errors.
    https://matplotlib.org/stable/gallery/statistics/boxplot_vs_violin.html

    Parameters
    ----------
    model : iterable
        Name of the used model:
            RW - Rescorla-Wagner
            H - RW-PH hybrid
            M - Meta learner
    dataList : list
        List containing the data filenames.
    datadir : Path
        Location of the data.
    wsls : bool, optional
        Set to True for plotting WSLS model. The default is False.

    Returns
    -------
    Plot.
    """

    models = af.labelDict()
    if not isinstance(model, str):
        model = [modeli.upper() for modeli in model]
    else:
        model = [model]

    # Plot specs
    fig, axs = plt.subplots(nrows=len(model), ncols=2,
                            figsize=(9, 4 * len(model)))
    fig.suptitle('Validity effect')
    labels = ['Valid trials', 'Invalid trials']

    for rowi, modeli in enumerate(model):
        # Wilhelm and Renee do not estimate PEs.
        assert not modeli in ['W', 'R', 'M'], 'Model has no estimated PE'

        # Preparation
        # -----------
        # Sort valid and invalid trials
        valid_pe = []
        invalid_pe = []
        for filei in dataList:
            data = pd.read_csv(datadir / filei, index_col='Unnamed: 0')
            valid_pe.append(list(data[f'RPE_{modeli}'].loc[data['validity'] == True]))
            invalid_pe.append(list(data[f'RPE_{modeli}'].loc[data['validity'] == False]))
        ## Reshape
        valid_long = [i for listi in valid_pe for i in listi]
        invalid_long = [i for listi in invalid_pe for i in listi]
        
        # Plot data
        pe_data = [valid_long, invalid_long]

        # Plot
        # ----
        ##SG: With the help of professor Dawyndt, Peter.
            # If there is more than 1 dimension, take the correct row.
            # Otherwise take everything.
        rawplots = axs[rowi, :] if axs.ndim > 1 else axs
        vplot, bplot = rawplots[0], rawplots[1]
        
        # Violin plot
        vplot.violinplot(pe_data,
                         showmeans=False,
                         showmedians=True)
        vplot.set_title(f'Violin plot {models[modeli]}')
        # Box plot
        bplot.boxplot(pe_data)
        bplot.set_title(f'Box plot {models[modeli]}')
        
        for ax in rawplots:
            ax.yaxis.grid(True)
            ax.set_xticks([y + 1 for y in range(len(pe_data))],
                          labels=labels)
            ax.set_xlabel('Trial type')
            ax.set_ylabel('Estimated prediction errors')

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
        List containing the data filenames.
    datadir : Path
        Location of the data.
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
            correct_Daphne = trial.relCue == trial.selCue_RW
            # correct_Daphne = trial.relCueCol == trial.selCue_RW
            DaphneList.append(correct_Daphne > 0.5)
            
            correct_Hugo = trial.relCue == trial.selCue_H
            # correct_Hugo = trial.relCueCol == trial.selCue_H
            HugoList.append(correct_Hugo > 0.5)
                
            correct_Wilhelm = trial.relCue == trial.selCue_W
            # correct_Wilhelm = trial.relCueCol == trial.selCue_W
            WilhelmList.append(correct_Wilhelm > 0.5)
            
            correct_Renee = trial.relCue == trial.selCue_R
            # correct_Renee = trial.relCueCol == trial.selCue_R
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
    plt.plot(meanLC[0, :], label='RW')
    plt.plot(meanLC[1, :], label='Hybrid')
    plt.plot(meanLC[3, :], label='Random')
    if wsls:
        ##SG: Makes the plot unclear.
        plt.plot(meanLC[2, :], label='WSLS')
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
        List containing the data filenames.
    datadir : Path
        Location of the data.
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


# ~~ Prediction error curve ~~ #
def pe_curve(model, dataList, datadir, plotnr, signed=True, peaks=False):
    """
    Plot prediction errors
    The prediction errors are plotted in a similar method as the proportion of
    choosing the predictive cue, trial by trial, depicted by the learning
    curves.

    Parameters
    ----------
    model : iterable
        Name of the used model:
            RW - Rescorla-Wagner
            H - RW-PH hybrid
            M - Meta learner
    dataList : list
        List containing the data filenames.
    datadir : Path
        Location of the data.
    plotnr : int
        Plot number.
    signed : bool, optional
        Plot signed or unsigned reward prediction errors.
        The default is True.
    peaks : bool, optional
        Highlight the peaks of prediction errors. The default is False.

    Returns
    -------
    Plot.
    """

    # Preparation
    # -----------
    models = af.labelDict()
    if not isinstance(model, str):
        model = [modeli.upper() for modeli in model]
    else:
        model = [model]
    
    ##SG: n models, 640 trials, n datasets
    LCmatrix = np.full((len(model), 640, len(dataList)), np.nan)
    for rowi, modeli in enumerate(model):
        # Wilhelm and Renee do not estimate PEs.
        assert not modeli in ['W', 'R', 'M'], 'Model has no estimated PE'
        
        for filei, file in enumerate(dataList):
            data = pd.read_csv(datadir / file, index_col='Unnamed: 0')
            
            # For each trial, is the predicitive cue selected,
            for triali, trial in data.iterrows():
                if signed:
                    LCmatrix[rowi, triali, filei] = trial[f'RPE_{modeli}']
                else:
                    LCmatrix[rowi, triali, filei] = abs(trial[f'RPE_{modeli}'])
    
    meanLC = np.nanmean(LCmatrix, axis=2)
    # Switch points
    ##SG: Only once since all data used the same experimental structure (at
        # least for the simulations -- fix this  when using participant data).
    lag_relCueCol = data.relCueCol.eq(data.relCueCol.shift())
    switches = np.where(lag_relCueCol == False)[0][1:]
    ## Lengt of the switch point bars
    barlen = (np.min(meanLC) + (np.min(meanLC) * 0.1),
              np.max(meanLC) + (np.max(meanLC) * 0.1))

    # Plot
    # ----
    plt.figure(plotnr)
    plt.xlabel('Trials')
    plt.ylabel('Mean prediction errors')
    plt.ylim(-1.1, 1.1)
    
    # Plot all participants and not the mean
    for rowi, modeli in enumerate(model):
        plt.plot(meanLC[rowi, :], label=models[modeli])
        if peaks:
            peakpoints, _ = signal.find_peaks(meanLC[rowi, :])
            plt.plot(peakpoints, meanLC[rowi, :][peakpoints], '-x',
                      label=f'peaks {models[modeli]}')
    plt.vlines(switches, barlen[0], barlen[1], colors='black')
    
    plt.legend()
    plt.tight_layout()
    plt.suptitle(
        f'Learning curve of prediction error over {len(dataList)} simulations',
        y=.99)

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


def heatmap_1d(model, plotdata, minima, tickvalues, truevalue=None):
    """
    1D heatmap
    Plot a heatmap and line graph for a 1 dimensional array.
    https://stackoverflow.com/a/45842334
    https://stackoverflow.com/a/8409110

    Parameters
    ----------
    model : iterable
        Name of the used model:
            RW - Rescorla-Wagner
            H - RW-PH hybrid
    plotdata : np.darray
        Data to be plotted.
    minima : np.darray
        Location of lowest values.
    tickvalues : np.darray
        Values of x-ax. For clarity, only a fourth will be presented.
    plotnr : int
        Plot number.

    Returns
    -------
    Plot.
    """

    # Preparation
    # -----------
    # Set title
    model = model.upper()
    models = af.labelDict()
    title = f'Negative Spearman Correlation {models[model]}'
    if model[:2] == 'RW':
        mdl_par = '$\u03B1$'
    else:
        mdl_par = '$\u03B7$'
    if not truevalue is None:
        title += f' ({mdl_par} = {truevalue})'
    # Size bounding box
    extent = [tickvalues[0] - (tickvalues[1] - tickvalues[0]) / 2,
              tickvalues[-1] + (tickvalues[1] - tickvalues[0]) / 2,
              0, 1]

    # Plot
    # ----
    fig, (hplot, lplot) = plt.subplots(nrows=2, sharex=True, figsize=(5, 2))
    # Heatmap
    hplot.imshow(plotdata[np.newaxis, :], cmap='plasma', aspect='auto',
                 extent=extent)
    hplot.set_xlim(tickvalues[0], tickvalues[-1])
    ## No need for values on this y-ax.
    hplot.set_yticks([])
    # Line graph
    lplot.plot(tickvalues, plotdata, marker='D', markevery=minima)
    lplot.set_xlabel(f'{mdl_par} values')
    lplot.set_xticks(np.round(tickvalues[::4], 3))
    lplot.set_ylabel('Correlation')
    
    plt.tight_layout()
    plt.suptitle(title)

    plt.show()



# ------------------------------------------------------------------------ End
