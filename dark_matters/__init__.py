"""
A code to calculate multi-frequency and multi-messenger emissions from WIMP annihilation and decay. This can be done both for standard channels and custom models, with the ability to produce surface brightnesses and integrated fluxes as well as maps in fits format to compare to actual data. This code makes use of an accelerated ADI solver (like Galprop) for electron diffusion with an innovative sparse matrix approach. Additionally, there is the option to use a Green's function approximate solution (implemented in both C++ and python).
"""
