# selfplot
selfplot is a package for visualizing numerical data with good defaults set for each plot.

During statistical analysis of data, a data scientist often needs to build various kinds of plots as he hopes to explore or confirm a hypothesis regarding the data. It can sometimes be a hurdle to write multiple lines of code before actually seeing a plot. In such cases, it is practical to have a minimum time delay between the conception of the hypothesis and the visualization of data.

The following package is an attempt to minimize the time taken to make visualizations. Reasonable defaults are set for each plots; however, they can be changed by the user if required.

To use the package, copy selfplot.py to the Lib\site-packages folder in the python installation folder. 
If you have Anaconda installed, copy selfplot.py to Anaconda\Lib\site-packages\ folder.

To Use the library in jupyter notebook/Spyder:

import selfplot    
selfplot.univariate(X)

