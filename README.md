# DarkMatters
A code to calculate multi-frequency and multi-messenger emissions from WIMP annihilations. This can be done both for standard channels and custom models, with the ability to produce surface brightnesses and integrated fluxes as well as flux maps in fits format to compare to actual data. Currently, this code makes use of the Green's function approximate solution for electron diffusion, this wil be updated in the near future with a full Crank-Nicolson approach.

## Requirements:
You will need a c++ compiler (g++ recommended). In addition, you require python3 with the numpy, scipy, matplotlib, pyfits, pyyaml and astropy packages installed.

## Installation:
Clone the repo to your local machine. 

For Linux, run "make" in the emissions sub-folder and you're done, yay Linux!

For Windows (you poor fool), try and use [CodeBlocks](http://www.codeblocks.org/downloads/) to compile emissions/electron.c (the compiler flags can be found in the .bat file in the same folder). To run the example notebook on Windows make sure you set the following variable in calculation.yaml: electronExecFile (set it to the location of executable you compiled from electron.c).

## Help:
View the [Wiki](https://github.com/Hyperthetical/DarkMatters/wiki) for help