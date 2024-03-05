# Theoretical_IICR_generator
Extension of the code of Willy (https://github.com/willyrv/IICREstimator/tree/master) for estimating IICR

## Setup
Requirements
1. For ms-based IICR, you need to install ms: https://snoweye.github.io/phyclust/document/msdoc.pdf
2. Working version of Python with the numpy library, my advice is to use Anaconda: https://numpy.org/install/
3. Python scripts estimIICR.py, model.py, and model_ssc.py

You need to have the ms files in the same folder as the Python scripts.

## Parameter/configuration file
For the ms-based simulations, see the tutorial on Willy's Github: https://github.com/willyrv/IICREstimator/blob/master/tutorial_simulating_IICR.md
### Computation of theoretical IICR
Stationary n-island models are defined in the "theoretical_IICR_nisland" section of the .json file:
+ "n": the number of islands
+ "M": the migration rate between islands
+ "sampling_same_island": 1 if we sample in the same island, 0 in different islands
  

All the other models are defined in the "theoretical_IICR_general" section of the .json file:
+ "M": list of migration matrices of the model (only one matrix if the model is stationary)
+ "tau": list of the times of changes of the model (if the model is stationary, put [0])
+ "sampling": sampling scheme in the present
+ "size":relative sizes of the demes

Every theoretical IICR has graphical parameters:
+ "label": name of the curve that appears in the legend of the plot
+ "color": color of the line representing the IICR (https://matplotlib.org/stable/gallery/color/named_colors.html)
+ "linestyle": style of the line representing the IICR (https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html)
+ "linewidth": thickness of the line representing the IICR (by default = 1.5)
+ "alpha": transparency of the line representing the IICR (by default = 1, for more transparent plot < 1)

### Other parameters
You can plot piecewise-constant functions (for example to show the real population size in your graph) using the "piecewise_constant_functions" section:
+ "x": 1D sequence of x positions. It is assumed, but not checked, that it is uniformly increasing
+ "y": 1D sequence of y levels
+ "label": "Real population size"
The y value is continued constantly to the right from every x position, i.e. the interval [x[i], x[i+1]) has the value y[i] (see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.step.html for more information, we use step function with where='post').

Important scale parameters are in the "scale_params" section:
+ "N0": Initial population size
+ "generation_time": Generation time of your specie

General parameters of the plot are found in the "plot_params" section:
+ "plot_theor_IICR": always put to 1 to compute theoretical IICR
+ "plot_real_ms_history": not interesting in our case
+ "plot_limits": list of four elements, the two first for the lower and upper limit of x-axis, and the two others for the lower and upper limit of y-axis
+ "plot_xlabel": label of x-axis
+ "plot_ylabel": label of y-axis

You can add vertical lines to the plot (for example to show time of changes in a non-stationary model) with the "vertical_lines" section, simply put a list of the x-values at which you want to draw a vertical line.

## Running the script to compute a theoretical IICR
1. Edit the parameter/configuration file in .json format
2. Run the command : python3 ./estimIICR.py parameters_configuration.json
   where parameters_configuration.json is the parameter/configuration file (you can name it as you want of course)
**WARNING ! This version of the code automatically create a test.png file in which the IICR are plotted. It will be erased if you run the code again with a different parameter file**
