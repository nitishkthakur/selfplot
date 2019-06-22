# selfplot
### selfplot is a python package for visualizing numerical data for Exploratory and Confirmatory data analysis.

During statistical analysis of data, a data scientist often needs to build various kinds of plots to explore or confirm a hypothesis regarding the data. It can sometimes be a hurdle to write multiple lines of code before actually seeing a plot. In such cases, it is practical to minimize the time delay between the conception of the hypothesis and the visualization of data.

The following python package is an attempt to minimize the time taken to make visualizations. Reasonable defaults are set for each plot; however, they can be changed manually if required. The package uses standard matplotlib and seaborn plotting functions. Visualizations are shown on the boston housing dataset. 

## Description
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

After copying the file, the package can be imported in jupyter notebook/Spyder:

```python
import selfplot    

# Univariate plot - Can be used to visualize skew in data and detect outliers.
# The histogram and the boxplot are positioned to make it easy to see which points on the histogram are outliers.
selfplot.univariate(X)
```
![Univariate Plot](https://github.com/nitishkthakur/selfplot/blob/master/Univariate.png?raw=true "Title")



```python
# Univariate time series plot - Time series plot of data and visualization of its distribution using Violin plot
selfplot.ts_univariate(ts_median_housing_prices)
```
![Univariate Time Series Plot](https://github.com/nitishkthakur/selfplot/blob/master/ts_Univariate.png?raw=true "Title")



```python
# Bivariate Binning plots - Can be used to visualize one variable as a function of another.
# The variable on x axis is divided into user-defined bins. 
# The boxplot shows the distribution of the y variable in each bin of the x variable
selfplot.bivariate_binning(data = data, x = 'LSTAT', y = 'median_housing_price', bins = range(1, 30, 5))
```
![Bivariate Binning Plot](https://github.com/nitishkthakur/selfplot/blob/master/binning_bivariate.png?raw=true "Title")

For more examples of using selfplot on data please view 'selfplot Examples.html'.

