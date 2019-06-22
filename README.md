# selfplot
### selfplot is a package for visualizing numerical data with defaults set for each plot.

During statistical analysis of data, a data scientist often needs to build various kinds of plots as he hopes to explore or confirm a hypothesis regarding the data. It can sometimes be a hurdle to write multiple lines of code before actually seeing a plot. In such cases, it is practical to minimize the time delay between the conception of the hypothesis and the visualization of data.

The following package is an attempt to minimize the time taken to make visualizations. Reasonable defaults are set for each plots; however, they can be changed manually if required.

<br> </br>

Following is a list and a brief description of the plots implemented in selfplot. For Examples of the plots on data please view selfplot Examples.html:

**univariate()** - Plot a histogram and a boxplot of data. Can be used to detect skew in data and the corresponding number of outliers through the boxplot.    
**ts_univariate()** - Plot a time series plot along with a violin plot to visualize the distribution.    
**bivariate_binning()** - Plot a variable y as a function of bins of variable x and plot the boxplot of y for every bin in x. This is similar to a scatterplot except that one of the variables is binned. The bins of x can be user defined.    
**histogram()** - Create histogram of data    
**line()** - Create Line plot or time series plot    
**kde()** - Plot the kernel density estimate    
**box()** - Plot the boxplot of a dataframe    
**violin()** - Plot a violin plot of a dataframe    
**scatter()** - Plot a scatter plot    
**hexbin()** - Plot a hexbin plot    
**heatmap()** - Plot a heatmap. This can be useful for visualizing correlations.    

To use the package, copy selfplot.py to the Lib\site-packages folder in the python installation folder. 
If you have Anaconda installed, copy selfplot.py to Anaconda\Lib\site-packages\

To Use the library in jupyter notebook/Spyder:

```python
import selfplot    
selfplot.univariate(X)
```

For Examples of selfplot on data please view selfplot Examples.html.

