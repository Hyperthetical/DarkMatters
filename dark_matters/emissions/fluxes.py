import numpy as np 
from scipy.integrate import simps as integrate
from scipy.interpolate import interp1d,interp2d
from astropy import constants as const
import warnings

def surfaceBrightnessLoopOld(nu_sb,fSample,rSample,emm,deltaOmega=4*np.pi):
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
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")

    n = len(rSample)
    sb = np.zeros(n,dtype=float) #surface brightness (nu,r)
    emm = interp2d(rSample,fSample,emm)
    
    for j in range(0,n-1):
        rprime = rSample[j]
        lSet = np.logspace(np.log10(rSample[0]),np.log10(np.sqrt(rSample[-1]**2-rprime**2)),num=3*n)
        rSet = np.sqrt(lSet**2+rprime**2)
        sb[j] = 2.0*integrate(emm(rSet,nu_sb),lSet)
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

def fluxFromJFactor(mx,z,jFactor,fSample,gSample,qSample,mode_exp):
    """
    High energy emmisivity from direct gamma-rays via J-factor
        ---------------------------
        Parameters
        ---------------------------
        halo - Required : halo environment (halo_env)
        phys - Required : physical environment (physical_env)
        sim  - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        2D float array (sim.num x sim.n) [cm^-2 s^-1]
    """
    num = len(fSample)
    h = const.h.to('GeV s').value
    me = (const.m_e*const.c**2).to('GeV').value #electron mass (GeV)
    nwimp0 = 0.25/np.pi/mx**mode_exp/mode_exp #GeV^-2
    emm = np.zeros(num,dtype=float)
    Q_func = interp1d(gSample,qSample)
    for i in range(0,num):
        E_g = h*fSample[i]*1e6*(1+z)/me
        if E_g < gSample[0] or E_g > gSample[-1]:
            emm[i] = 0.0
        else:
            emm[i] = Q_func(E_g)*jFactor*nwimp0*E_g #units of flux
            #print halo.J,Q_func(E_g)*nwimp0*E_g/(1+halo.z),1+halo.z
    return 2.0*emm #2 gamma-rays per event 