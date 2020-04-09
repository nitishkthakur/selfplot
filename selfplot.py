import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn import metrics

def help():
    print('''Functions available in selfplot:
    univariate() - Plot a histogram and a boxplot of data. Can be used to detect skew in data and the corresponding number of outliers through the boxplot.
    ts_univariate() - Plot a time series plot along with a violin plot to visualize the distribution.
    bivariate_binning() - Plot a variable y as a function of bins of variable x and plot the boxplot of y for every bin in x. This is similar to a scatterplot except that one of the variables is binned. The bins of x can be user defined.
    histogram() - Create histogram of data
    line() - Create Line plot or time series plot
    kde() - Plot the kernel density estimate
    box() - Plot the boxplot of a dataframe
    violin() - Plot a violin plot of a dataframe
    scatter() - Plot a scatter plot
    hexbin() - Plot a hexbin plot
    heatmap() - Plot a heatmap. This can be useful for visualizing correlations.
    bar() - Make an bar plot with annotation''')
    
    

# Set Default Theme
plt.style.use('seaborn-whitegrid')

def histogram(x, bins = 14, density = False, color = 'teal', figsize = (8,3.5), dpi = 150, showmean = False, fontsize = 9):
    '''
    x: variable to be binned
    bins: Number of bin or values of Bin edges
    density: if False, then Frequency is plotted on the y axis. If True, density is plotted on the y axis.
    color: color of the histogram bars.
    figsize: figure size. Default = (8, 3.5)
    dpi: dots per inch
    showmean: if True, a vertical line is plotted which shows the mean.
    fontsize: fontsize for ticks and labels'''
    if isinstance(x, pd.core.series.Series):
        xlabel = x.name
        x = x.dropna()    
    else:
        xlabel ='Parameter Value'
        x = x[~np.isnan(x)]
    fig = plt.figure(figsize=figsize, dpi = dpi)
    n, bins, patches = plt.hist(x, rwidth=.96, alpha=.93, color=color, bins = bins, density=density)
    plt.xticks(bins, np.around(bins, 2), fontsize = fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize+1)
    if density == False:
        plt.ylabel('Frequency', fontsize=fontsize+1)
    if density == True:
        plt.ylabel('Density', fontsize = fontsize+1)
    if showmean == True:
        plt.axvline(x = x.mean(), linestyle = '--', color='black')
        ymin, ymax = plt.gca().get_ylim()
        plt.annotate(s = 'Mean='+str(np.around(x.mean(),2)), xy = (x.mean(), .75*(ymax-ymin)),
             xytext = (x.mean(), .75*(ymax-ymin)), color='black', fontsize=fontsize)
    return plt
    
    
    
def line(x, figsize=(8, 3.5), fontsize = 9, marker = None, linewidth = 1, dpi =150, markersize = 3):
    '''
    x: variable to be plotted
    figsize: figure size
    fontsize: fontsize of ticks and labels
    marker: marker as defined in matplotlib. For Example, assign 'o' for circular markers.
    linewidth: width of line as defined in matplotlib. default value = 1
    dpi: dots per inch. default value = 150
    markersize: Size of marker(if any)'''
    if isinstance(x, pd.core.series.Series):
        ylabel = x.name
        x = x.dropna()    
    else:
        ylabel ='Parameter Value'
        x = x[~np.isnan(x)]
    fig = plt.figure(figsize=figsize, dpi = dpi)
    plt.plot(x, marker = marker, linewidth = linewidth, markersize =markersize)
    
    # set labels
    plt.ylabel(ylabel, fontsize = fontsize+1)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    return plt
    
    
def kde(x, shade = True, figsize=(8,3.5), dpi = 150, fontsize =12):
    '''x: data to be plotted
    shade: If True, area under the curve will be shaded.
    fontsize: set fontsize of labels and ticks.
    figsize: figure size
    dpi: dots per inch'''
    if isinstance(x, pd.core.series.Series):
        xlabel = x.name
    else:
        xlabel =''
    fig = plt.figure(figsize=figsize, dpi = dpi)
    sns.kdeplot(x, shade = shade)
    
    # set labels
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel('Density', fontsize=fontsize)
    
    # Give custom xticks
    xmin, xmax = plt.gca().get_xlim()
    plt.xticks(np.linspace(xmin, xmax, 10), np.around(np.linspace(xmin, xmax, 15),2))
    plt.legend(frameon =True)
    plt.grid(alpha = .6)
    return plt
    
def box(df, colnames = '', notch = False, figsize = (3,4), dpi = 150, fontsize = 9):
    '''df: Input dataframe
    colnames: List of column names of df which are to be plotted.
    notch: Notch for uncertainty quantification of median.
    fontsize: set fontsize of labels and ticks.
    figsize: figure size
    dpi: dots per inch'''
    if list(colnames) != list(''):
        labels = list(colnames)
     
    # Decide on Figure Size
    figure_set = False
    if figsize != (3,4):
        figure_set = True
    
    if (len(colnames) > 1)&(figure_set == False):
        figsize = (len(colnames), 3)
    
    # Create the Boxplot
    data = np.array(df[labels])
    plt.figure(figsize=figsize, dpi = dpi)
    plt.boxplot(data, labels = labels, notch=notch)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize =fontsize)
    return plt


def violin(df, colnames = '', figsize = (3,4), dpi = 150, fontsize = 9):
    '''df: Input dataframe
    colnames: List of column names of df which are to be plotted.
    fontsize: set fontsize of labels and ticks.
    figsize: figure size
    dpi: dots per inch'''
    if list(colnames) != list(''):
        labels = list(colnames)
     
    # Decide on Figure Size
    figure_set = False
    if figsize != (3,4):
        figure_set = True
    
    if (len(colnames) > 1)&(figure_set == False):
        figsize = (len(colnames), 3)
    
    # Create the Boxplot
    data = np.array(df[labels])
    plt.figure(figsize=figsize, dpi = dpi)
    plt.violinplot(data, showmedians = True)
    plt.xticks(range(1, len(labels) + 1), labels, fontsize = fontsize)
    plt.yticks(fontsize =fontsize)
    return plt


def scatter(x, y, s = 19, c = 'teal', figsize = (6,4), cmap = 'Blues', dpi = 150, alpha =.9):
    '''
        x: x axis data to be plotted.
    y: y axis data to be plotted.
    s = size of marker; Can set it equal to a variable to make the size a function of the variable value.
    c: color of marker. Can set it equal to a variable to make the color a function of the variable value.
    alpha: Opacity of markers.
    cmap: color map for frequency count.
    gridsize: number of hexbins in the range of each axis.
    figsize: figure size.
    dpi: dots per inch.
    '''
    if isinstance(x, pd.core.series.Series):
        xlabel = x.name
    else:
        xlabel = 'X'
    
    if isinstance(y, pd.core.series.Series):
        ylabel = y.name
    else:
        ylabel = 'Y'
    if isinstance(c, pd.core.series.Series):
        clabel = c.name
    else:
        clabel = 'Z'
    plt.figure(figsize = figsize, dpi = dpi)
    plt.scatter(x, y, s = s, c = c, cmap = cmap, alpha = alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if isinstance(c, str): 
        pass
    else:
        colorbar_plot = plt.colorbar()
        colorbar_plot.set_label(clabel)
    return plt

def hexbin(x, y, figsize = (7,5), cmap = 'Blues', gridsize = 20, C = None, dpi = 150):
    '''
    x: x axis data to be plotted
    y: y axis data to be plotted
    cmap: color map for frequency count
    gridsize: number of hexbins in the range of each axis
    figsize: figure size
    dpi: dots per inch
    '''
    if isinstance(x, pd.core.series.Series):
        xlabel = x.name
    else:
        xlabel = 'X'

    if isinstance(y, pd.core.series.Series):
        y = y.dropna()
        ylabel = y.name
    else:
        ylabel = 'Y'
    
    plt.figure(figsize = figsize, dpi = dpi)
    plt.hexbin(x, y, C = C, cmap = cmap, gridsize = gridsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label = 'Frequency')
    plt.grid(False)
    return plt

def ts_univariate(x, figsize = (13,4), dpi =150, marker = None, markersize = None, fontsize = 12):
    '''
    Generates a Composite Plot for Quick univariate visualization of time series data.
    x: data to be plotted
    figsize: figure size
    dpi: dots per inch
    marker: marker to be used in the plot
    markersize: markersize of the marker in time series plot
    fontsize: fontsize of ticks and labels'''
    if isinstance(x, pd.core.series.Series):
        xlabel = x.name
        if xlabel == None:
            xlabel = ''
    else:
        xlabel = ''

    fig = plt.figure(figsize = figsize, dpi = dpi)
    grid = gridspec.GridSpec(1,5)

    ax1 = fig.add_subplot(grid[:-1])
    ax1.plot(x, marker = marker, markersize = markersize, alpha = .85)
    ax1.set_ylabel(xlabel, weight = 'bold', fontsize = fontsize+1)

    ax2 = fig.add_subplot(grid[-1])
    ax2.violinplot(x, showmedians = True)
    ax2.set_xticklabels([])
    plt.setp(ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels(), fontsize = fontsize)
    return plt

def bivariate_binning(data, x, y, bins = 6, figsize = (8.5,4), dpi = 150, fontsize = 9, color ='steelblue', round_off = 2):
    '''
    data: dataframe containing x and y variables.
    x: string containing name of variable to be binned. This variable will be on the x-axis.
    y: string containing name of variable to be on the y-axis.
    bins: Number of bins required for x
    figsize: figure size. default: (6,4)
    dpi: dots per inch. default: 150
    fontsize: fontsize for labels and ticks
    color: color of the boxplot
    round_off: Number of places to round off x tick labels to. (Prevents overlapping of ticklabels)
    
    '''
    fig = plt.figure(figsize = figsize, dpi = dpi)
    
    data = data[[x,y]]
    data = data[[x,y]]

    
    if isinstance(bins, int) == True:
        bins = np.linspace(data[x].min(), data[x].max(), bins+1)
        
    xlabels = []
    data['bin_marker'] = 1
    for temp_bin_val in range(len(bins)-1):
        mask = (data[x]>=bins[temp_bin_val])&(data[x]<bins[temp_bin_val+1])
        data.loc[:,'bin_marker'].loc[mask] = temp_bin_val
        xlabels.append(str(np.around(bins[temp_bin_val], round_off)) + '-' + str(np.around(bins[temp_bin_val+1], round_off)))
        
    plt.figure(figsize = figsize, dpi = dpi)
    sns.boxplot(data = data, x = 'bin_marker', y = y, color = color)
    plt.xlabel(x, fontsize = fontsize, weight ='bold')
    plt.ylabel(y, fontsize = fontsize, weight = 'bold')
    plt.xticks(range(0, len(xlabels)), xlabels)
    return plt
    
    
def heatmap(df, figsize = None, dpi = 150, cmap = 'coolwarm', vmax = None, vmin =None, annot = True, 
            annot_kws = {'fontsize': 11}, linewidth = .1, fontsize = 12, cbar_fontsize = 13):
    '''df: dataframe input
    figsize: figure size; if not given then it is calculated automatically
    dpi: dpi value
    cmap: colormap; recommended options: bipolar = "coolwarm", One color = "OrRd" 
    annot: True to annotate the cells
    annot_kws: annotation fontsize, fontname to be set
    linewidth: linewidth separating neighboring squares
    fontsize: x and y ticks size; default: 12
    cbar_fontsize: colorbar fontsize; default: 13'''
    if type(figsize) == type(None):
        figsize = (df.shape[1]*.99, df.shape[0]*.85)
    fig = plt.figure(figsize = figsize, dpi = dpi)
    ax = sns.heatmap(df, cmap =cmap, vmax = vmax, vmin = vmin, annot = annot, annot_kws = annot_kws, linewidth = linewidth)
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    plt.yticks(rotation = 0, fontsize = fontsize)
    plt.xticks(rotation = 90, fontsize = fontsize)
    return plt
    
    
def univariate(x, bins = 12, figsize = (13,7), dpi =150, fontsize = 13, round_off = 2, showmeans = False):
    '''
    Generates a Composite Plot for Quick univariate visualization of time series data.
    x: data to be plotted
    figsize: figure size
    dpi: dots per inch
    fontsize: fontsize of ticks and labels,
    round_off: Number of decimal places to roundoff the histogram x ticks to.
    showmeans: True to mark the mean value in the boxplot'''
    if isinstance(x, pd.core.series.Series):
        xlabel = x.name
        if xlabel == None:
            xlabel = ''
    else:
        xlabel = ''

    fig, ax = plt.subplots(2,1,figsize = figsize, dpi = dpi)
    ax1 = ax[0]
    ax2 = ax[1]
    
    n, bins, patches = ax1.hist(x, color = 'teal', rwidth = .96, alpha = .85, bins = bins)
    ax1.set_xticks(bins)
    ax1.set_xticklabels(np.around(bins, round_off))
    ax1.set_ylabel('Frequency', weight = 'bold', fontsize = fontsize+1)
    
    ax2.boxplot(np.array(x), showmeans = showmeans, vert = False)
    ax2.set_ylabel(xlabel, fontsize = fontsize + 1, weight = 'bold')
    plt.setp(ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels()+ ax2.get_xticklabels(), fontsize = fontsize)
    return plt
    
    
def bar(x, y, annotate = True, fontsize = 10, annot_x_offset = -.25, annot_y_offset = .02, annot_weight = None,
        annot_fontsize = 10, figsize = (7, 4), dpi = 120, xlabel = None, ylabel = None, title = None):
    '''x: x axis labels passed as array or series(can be string values)
    y: heights of the bars
    annot_x_offset: horizontal position of the annotation text relative to the center of the bar, default: -0.25
    annot_y_offset: vertical space between annotation and top of the bar measured relative to minimum y value seen in data. default: .02
    annot_weight: if set to "bold", the annotation text sppears in bold, if set to None, no weight added
    annot_fontsize: fontsize of annotation text
    figsize: figure size
    dpi: dpi of figure, default:120'''
    fig = plt.figure(figsize = figsize, dpi = dpi)
    xcount = range(len(x))
    ymin = min(y)
    
    plt.bar(x = xcount, height = y, alpha = .92)
    plt.xticks(xcount, x, weight = 'bold')
    plt.yticks(weight = 'bold')
    plt.grid(alpha = .5)
    if xlabel: plt.xlabel(xlabel, fontsize = fontsize)
    if ylabel: plt.ylabel(ylabel, fontsize = fontsize)
    if title: plt.title(title, fontsize = fontsize)
    if annotate:
        for counter in xcount:
            if y[counter] >= 0:
                plt.text(counter+annot_x_offset, y = y[counter]+(ymin*annot_y_offset), s = str(y[counter]), weight = annot_weight, fontsize = annot_fontsize)
            else:
                plt.text(counter+annot_x_offset, y = y[counter]+(ymin*annot_y_offset*2), s = str(y[counter]), weight = annot_weight, fontsize = annot_fontsize)
    return plt
    
# Visualize Model Output
def prediction_diagnose(y_true, y_pred, figsize = (15, 10), dpi = 120, fontsize = 14, varname = 'Y', metrics_display = True, rounded = 3):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    r2 = np.around(metrics.r2_score(y_true, y_pred), rounded)
    rmse = np.around(np.sqrt(metrics.mean_squared_error(y_true, y_pred)), rounded)
    mae = np.around(metrics.mean_absolute_error(y_true, y_pred), rounded)
    
    fig = plt.figure(figsize = figsize, dpi = 120, tight_layout = True)
    grid = gridspec.GridSpec(2,4)
    
    ax1 = fig.add_subplot(grid[0, :])
    ax1.plot(y_true, color = 'teal', label = 'True Value', alpha = .87)
    ax1.plot(y_pred, color = 'tomato', label = 'Predicted Value', alpha = .87)
    ax1.legend(frameon = True, fontsize = fontsize - 1)
    ax1.set_ylabel(varname, fontsize = fontsize)
    if metrics_display: ax1.set_title('R2: {}, RMSE: {}, MAE: {}'.format(r2, rmse, mae), fontsize = fontsize + 2)
    plt.setp(ax1.get_xticklabels()+ax1.get_yticklabels(), fontsize = fontsize)
    
    ax2 = fig.add_subplot(grid[1, :3])
    sns.distplot(y_true, label = 'True Value', color = 'teal', ax = ax2)
    sns.distplot(y_pred, label = 'Predicted Value', color = 'tomato', ax = ax2)
    plt.setp(ax2.get_xticklabels()+ax2.get_yticklabels(), fontsize = fontsize)
    ax2.set_xlabel(varname, fontsize = fontsize)
    ax2.legend(frameon = True, fontsize = fontsize - 1)
    
    ax3 = fig.add_subplot(grid[1, 3:])
    ax3.boxplot(np.concatenate([y_true[:, None], y_pred[:, None]], axis = 1), showmeans = True)
    ax3.set_xticklabels(['True', 'Predicted'])
    plt.setp(ax3.get_xticklabels()+ax3.get_yticklabels(), fontsize = fontsize)
    return plt

# Visualize Behaviour of residuals
def residual_diagnose(y_true, y_pred, figsize = (15, 10), dpi = 120, fontsize = 15, varname = 'Y', parity_delta = 5, 
                      line_zero = True):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    residuals = y_true - y_pred
    
    fig = plt.figure(figsize = figsize, dpi = 120, tight_layout = True)
    grid = gridspec.GridSpec(2,2)
    
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.plot(y_pred, residuals, color = 'teal', alpha = .87, linewidth = 0, marker = 'o')
    ax1.set_ylabel('Residuals', fontsize = fontsize)
    ax1.set_xlabel('Predicted Y', fontsize = fontsize)
    if line_zero: ax1.axhline(0, color = 'black')
    plt.setp(ax1.get_xticklabels()+ax1.get_yticklabels(), fontsize = fontsize)
    
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.plot(y_true, residuals, color = 'teal', alpha = .87, linewidth = 0, marker = 'o')
    plt.setp(ax2.get_xticklabels()+ax2.get_yticklabels(), fontsize = fontsize)
    ax2.set_xlabel("True Y", fontsize = fontsize)
    ax2.set_ylabel('Residuals', fontsize = fontsize)
    if line_zero: ax2.axhline(0, color = 'black')
    ax2.legend(frameon = True, fontsize = fontsize - 1)
    
    ax3 = fig.add_subplot(grid[1, 0])
    ax3.plot(y_true, y_pred, linewidth = 0, marker = 'o', alpha = .87)
    ax3.plot(y_true, y_true*(100+parity_delta)/100, color = 'tomato', alpha = .85)
    handle = ax3.plot(y_true, y_true*(100-parity_delta)/100, color = 'tomato', alpha = .85)
    ax3.legend(handle, ['+-'+str( parity_delta)+'%'], fontsize = fontsize, frameon = True)
    ax3.set_xlabel('True Y', fontsize = fontsize)
    ax3.set_ylabel('Predicted Y', fontsize = fontsize)
    plt.setp(ax3.get_xticklabels()+ax3.get_yticklabels(), fontsize = fontsize)
    
    ax4 = fig.add_subplot(grid[1, 1])
    sns.distplot(residuals)
    ax4.set_xlabel('Residuals', fontsize = fontsize)
    plt.setp(ax4.get_xticklabels()+ax4.get_yticklabels(), fontsize = fontsize)
    return plt
    