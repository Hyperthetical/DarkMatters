# DarkMatters
Setup:

Linux: run make in the emissions folder

Windows: you need to make this work, there is a sample compilation bat file

To run the example notebook make sure you set the following variable in calculation.yaml: electronExecFile

More detail can be found in the [Wiki](https://github.com/Hyperthetical/DarkMatters/wiki)

Some input variables can have a unit property. These should be included in astropy readable format. To see the unit lists take a look at the config folder and the quantities.yaml file. 

calcData:

calcRmax: either a value or -1/"Rmax" without units, the latter give Rmax = Rvir