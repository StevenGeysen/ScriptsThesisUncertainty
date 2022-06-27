#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Plot functions -- Version 1
Last edit:  2022/06/27
Author(s):  Geysen, Steven (SG)
Notes:      - Functions used to plot output
                * Heatmap
            - Release notes:
                * Heatmap
            
To do:      - Implement in other scripts
            - Add functions of other often used plots
            
Comments:   
            
Sources:     https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
"""



#%% ~~ Imports ~~ %%#


import matplotlib

import numpy as np
import matplotlib.pyplot as plt



#%% ~~ Heatmap ~~ %%#
#-------------------#


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



# ------------------------------------------------------------------------ End
