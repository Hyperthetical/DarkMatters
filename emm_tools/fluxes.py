import numpy as np 
from scipy.integrate import simps as integrate
from scipy.interpolate import interp1d,interp2d

def surfaceBrightnessLoop(nu_sb,fSample,rSample,emm,deltaOmega=4*np.pi):
    """
    Surface brightness from emmissivity 
        ---------------------------
        Parameters
        ---------------------------
        nu          - Required : frequency value for calculation (float) [MHz]
        rvir        - Required : virial radius of target (float) [Mpc]
        fSample     - Required : frequency points for calculation (1d array-like length n) [MHz]
        rSample     - Required : radial points for calculation (1d array-like length m) [Mpc]
        emm         - Required : emmissivity (2d array-like, dimension (n,m)) []
        deltaOmega  - Optional : angular area of flux distribution (float) [sr]
        ---------------------------
        Output
        ---------------------------
        1D float array of surface-brightness (length n) [Jy arcminute^-2]
    """
    n = len(rSample)
    lum = np.zeros(n,dtype=float)
    sb = np.zeros(n,dtype=float) #surface brightness (nu,r)
    if nu_sb in fSample:
        emm_nu = emm[fSample==nu_sb][0]
    else:
        emm = interp2d(rSample,fSample,emm)
        emm_nu = emm(rSample,nu_sb)
    if any(np.isnan(emm_nu)):
        nanIndex = np.abs(np.where(np.isnan(emm_nu))[0][0] - n)
    else:
        nanIndex = 0
    for j in range(0,n-nanIndex):
        rprime = rSample[j]
        for k in range(0,n-nanIndex):    
            r = rSample[k]    
            if(rprime >= r):
                lum[k] = 0.0
            else:
                lum[k] = emm_nu[k]*r/np.sqrt(r**2-rprime**2)
        sb[j] = 2.0*integrate(lum,rSample) #the 2 comes from integrating over diameter not radius
    deltaOmega *= 1.1818e7 #convert sr to arcminute^2
    return rSample,sb*3.09e24*1.6e20/deltaOmega #unit conversions and adjustment to angles 

def fluxGrid(rf,dl,fSample,rSample,emm,boostMod=1.0):
    """
    Radio flux from emmissivity 
        ---------------------------
        Parameters
        ---------------------------
        rf          - Required : radial limit of flux integration (float) [Mpc]
        dl          - Required : luminosity distance of target (float) [Mpc]
        fSample     - Required : frequency points for calculation (1d array-like length n) [MHz]
        rSample     - Required : radial points for calculation (1d array-like length m) [Mpc]
        emm         - Required : emmissivity (2d array-like, dimension (n,m)) []
        boost       - Optional : flux boost factor (float) []
        radio_boost - Optional : radio flux boost factor (float) []
        ---------------------------
        Output
        ---------------------------
        1D float array of fluxes (length n) [Jy]
    """
    emmInt = interp2d(rSample,fSample,emm)
    newRSample = np.logspace(np.log10(rSample[0]),np.log10(rf),num=100)
    fGrid,rGrid = np.meshgrid(fSample,newRSample,indexing="ij")
    fluxGrid = rGrid**2*emmInt(newRSample,fSample)/(dl**2 + rGrid**2)
    return integrate(fluxGrid,rGrid)*3.09e24*1.60e20*boostMod

def fluxLoop(rf,dl,fSample,rSample,emm,boostMod=1.0):
    """
    Radio flux from emmissivity 
        ---------------------------
        Parameters
        ---------------------------
        rf          - Required : radial limit of flux integration (float) [Mpc]
        dl          - Required : luminosity distance of target (float) [Mpc]
        fSample     - Required : frequency points for calculation (1d array-like length n) [MHz]
        rSample     - Required : radial points for calculation (1d array-like length m) [Mpc]
        emm         - Required : synchrotron emmissivity (2d array-like, dimension (n,m)) []
        boost       - Optional : flux boost factor (float) []
        radio_boost - Optional : radio flux boost factor (float) []
        ---------------------------
        Output
        ---------------------------
        1D float array of fluxes (length n) [Jy]
    """
    n = len(rSample)
    num = len(fSample)
    jj = np.zeros(n,dtype=float)   #temporary integrand array
    ff = np.zeros(num,dtype=float)    #flux density
    for i in range(0,num):
        if rf < 0.9*rSample[n-1]:
            halo_interp = interp1d(rSample,emm[i])
            rset = np.logspace(np.log10(rSample[0]),np.log10(rf),num=n)
            emm_r = halo_interp(rset)
        else:
            emm_r = emm[i]
            rset = rSample
        jj = rset**2*emm_r
        #flux density as a function of frequency, integrate over r to get there
        ff[i] = 4.0*np.pi*integrate(jj/(dl**2+rset**2),rset)/(4.0*np.pi)
    ff = ff*3.09e24   #flux density in GeV cm^-2
    ff = ff*1.60e20    #flux density in Jy
    ff = ff*boostMod #accounts for reduced boost for radio flux when using Prada 2013 boosting
    #results must be multiplied by the chi-chi cross section
    return ff