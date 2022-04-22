# DarkMatters
A code to calculate multi-frequency and multi-messenger emissions from WIMP annihilations. This can be done both for standard channels and custom models, with the ability to produce surface brightnesses and integrated fluxes as well as flux maps in fits format to compare to actual data. This code makes use of a Crank-Nicolson accelerated ADI solver (like Galprop) for electron diffusion. Additionally, there is the option to use a Green's function approximate solution.

## Requirements:
This software requires python3 with the numpy, scipy, sympy, matplotlib, pyyaml, and astropy packages installed. A requirements.txt file is provided for use with pip.

### Green's function solutions
To use these you require a c++ compiler (g++ recommended).

## Installation:
Clone the repo to your local machine. 

### Green's function solutions
If you want to make use of the Green's function solution method you need to compile a c executable. By default DarkMatters uses a Crank-Nicolson solution, implemented in python (it is recommended to use this approach).

For Linux, run "make" in the emissions sub-folder and you're done, yay Linux!

For Windows (you poor fool), try and use [CodeBlocks](http://www.codeblocks.org/downloads/) to compile emissions/electron.c (the compiler flags can be found in the .bat file in the same folder). To run the example notebook on Windows make sure you set the following variable in calculation.yaml: electronExecFile (set it to the location of executable you compiled from electron.c).

## Help:
View the [Wiki](https://github.com/Hyperthetical/DarkMatters/wiki) for help