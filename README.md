# DarkMatters
A code to calculate multi-frequency and multi-messenger emissions from WIMP annihilation and decay. This can be done both for standard channels and custom models, with the ability to produce surface brightnesses and integrated fluxes as well as maps in fits format to compare to actual data. This code makes use of an accelerated ADI solver (like Galprop) for electron diffusion with an innovative sparse matrix approach. Additionally, there is the option to use a Green's function approximate solution (implemented in both C++ and python).

## Requirements:
This software requires python3 with the numpy, scipy, sympy, matplotlib, pyyaml, jupyter, tqdm, joblib, and astropy packages installed. A requirements.txt file is provided for use with pip.

### Green's function solutions
**These do not currently work as they are being reimplemented** On unix systems you can use these in their faster C++ implementation, this requires a C++ compiler (g++ recommended) and the Gnu Scientific Libray (GSL), particularly the libgsl-dev package.

## Installation:
Clone the repo to your local machine. 

### Green's function solutions
**These do not currently work as they are being reimplemented** If you want to make use of the C++ Green's function solution method you need to compile an executable.

On Linux, run "make" in the emissions sub-folder and you're done, yay Linux!

On Windows, you are out of luck (unless you can get a C++ compiler with GSL working), but you can use the python implementation.

## Help:
View the [Wiki](https://github.com/Hyperthetical/DarkMatters/wiki) for help

## Citation
If you use the generic WIMP channels please cite 

M.Cirelli, G.Corcella, A.Hektor, G.Hütsi, M.Kadastik, P.Panci, M.Raidal, F.Sala, A.Strumia, "PPPC 4 DM ID: A Poor Particle Physicist Cookbook for Dark Matter Indirect Detection", arXiv 1012.4515, JCAP 1103 (2011) 051. 

P. Ciafaloni, D. Comelli, A. Riotto, F. Sala, A. Strumia, A. Urbano, "Weak corrections are relevant for dark matter indirect detection",arXiv 1009.0224, JCAP 1103 (2011) 019.

If you use the 2HDM+S channel please cite

G. Beck, R. Temo, E. Malwa, M. Kumar, B. Mellado, "Connecting multi-lepton anomalies at the LHC and in Astrophysics with MeerKAT/SKA", arxiv:2102.10596, Astroparticle Physics, 148 (2023), 102821.