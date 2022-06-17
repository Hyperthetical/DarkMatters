# DarkMatters
A code to calculate multi-frequency and multi-messenger emissions from WIMP annihilations. This can be done both for standard channels and custom models, with the ability to produce surface brightnesses and integrated fluxes as well as flux maps in fits format to compare to actual data. This code makes use of an accelerated ADI solver (like Galprop) for electron diffusion with an innovative sparse matrix approach. Additionally, there is the option to use a Green's function approximate solution (implemented in both C++ and python).

## Requirements:
This software requires python3 with the numpy, scipy, sympy, matplotlib, pyyaml,jupyter,tqdm, joblib, and astropy packages installed. A requirements.txt file is provided for use with pip.

### Green's function solutions
On unix systems you can use these in their faster C++ implementation, this requires a c++ compiler (g++ recommended) and the Gnu Scientific Libray (GSL).

## Installation:
Clone the repo to your local machine. 

### Green's function solutions
If you want to make use of the C++ Green's function solution method you need to compile a c executable. By default DarkMatters uses an ADI solution, implemented in python (it is recommended to use this approach).

On Linux, run "make" in the emissions sub-folder and you're done, yay Linux!

On Windows, you are out of luck but you can use the python implementation.

## Help:
View the [Wiki](https://github.com/Hyperthetical/DarkMatters/wiki) for help