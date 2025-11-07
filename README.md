# Thesis Code

Repository containing all major code used throughout the completion of my MSc Environmental Modelling thesis 'Numerical modelling of the evolution of Alaska’s land-terminating glaciers until 2100'. 

All glacier change simulations with OGGM were run on my personal Macbook Air, hence the functions in [SimCode.py](https://github.com/johncort/ThesisCode/blob/main/Modules/SimCodes.py) were designed to perform simulations in chunks, preventing issues with overloading and slowdown that seemed to arise if too many glaciers were passed to OGGM in one go. A demonstration of the running of these simulation functions in a notebook can be found [here](https://github.com/johncort/ThesisCode/blob/main/NotebookDemos/SimulationDemo.ipynb), wherein I demonstrate the chunk process on a small sample of glaciers. In this notebook I also demonstrate the use of [OGGM's ice thickness redistribution function](https://tutorials.oggm.org/stable/notebooks/tutorials/distribute_flowline.html), rerunning the simulations for select glaciers before processing their ice thickness and exporting for analysis in QGIS. 

[OGGM v1.6.2 (latest release)](https://docs.oggm.org/en/latest/whats-new.html) was used for all simulations. For privacy, all filepaths have been replaced with 'custom_filepath'.
OGGM's documentation can be found [here](https://docs.oggm.org/en/stable/) and helpful code for OGGM can be found in a series of informative notebooks [here](https://tutorials.oggm.org/stable/notebooks/welcome.html).

Given the storage requirements necessary to perform these simulations on my personal computer, processing the output for hypsometric analyses required running the script solely in the terminal, making use of multiprocessing. This script can be found [here](https://github.com/johncort/ThesisCode/blob/main/Modules/Analysis/ProcessHypsometry.py), but its usage is not demonstrated. 

During the glacier simulations, the climate data used for forcing was additionally saved in order to create plots of the subregional variability in projected temperature and precipitation change. Across the sample of ~10,000 glaciers, each climate file was multiple GBs large, hence the [conversion of CSV files to Parquet files for their processing](https://github.com/johncort/ThesisCode/blob/main/Modules/Analysis/ProcessClimateData.py). 

<p align="center">
  <img width="5500" height="2000" alt="MultiVol" src="https://github.com/user-attachments/assets/57ec1292-f31b-4fef-ab4b-c1421459524a" /><br>
  <i>Simulated regional (a) and subregional (b) volume change of Alaska's land-terminating glaciers</i>
</p>

All analyses were conducted in Jupyter Notebooks, however I have compiled the data processing, analysis, and plotting into a series of functions which are demonstrated [here](https://github.com/johncort/ThesisCode/blob/main/NotebookDemos/Plotting/PlottingDemo.ipynb) - all thesis plots, including those created using QGIS can be found [here](https://github.com/johncort/ThesisCode/tree/main/NotebookDemos/Plotting/Plots), however please note that some of these received minor formatting changes when finalising the thesis that are not included in the uploaded code. Additionally, note that [Figure 4](https://github.com/johncort/ThesisCode/blob/main/NotebookDemos/Plotting/Plots/Figure4.png) was a last minute addition and since its workflow is adapted from an OGGM tutorial, I instead signpoint you to this [OGGM notebook](https://tutorials.oggm.org/stable/notebooks/tutorials/full_prepro_workflow.html). 

Whilst I have tried to make these functions as reusable as possible, their primary goal was to deliver the simulations, analyses, and plots required for my thesis, hence they would likely require tweaking for other uses. Arising following plenty of trial and error when dealing with such large datasets, much of the analysis and plotting code, whilst functional for my needs, is likely far from optimal. I have tried to provide expanded commenting for various lines of code to provide the rationale behind my decision making.

N.B. This repository recreates my original thesis repository but removes any mention of my candidate ID - the code files are unchanged, only some small restructuring of filepaths and reformatting of this README has been made. 
