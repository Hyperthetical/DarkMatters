import numpy as np 
from scipy.integrate import simps as integrate
from scipy.interpolate import interp1d,interp2d
from astropy import constants as const

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
    sb = np.zeros(n,dtype=float) #surface brightness (nu,r)
    emm = interp2d(rSample,fSample,emm)
    
    for j in range(0,n-1):
        rprime = rSample[j]
        lSet = np.logspace(np.log10(rSample[0]),np.log10(np.sqrt(rSample[-1]**2-rprime**2)),num=200)
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
    if len(rSample) < 200:
        newRSample = np.logspace(np.log10(rSample[0]),np.log10(rf),num=200)
    else:
        newRSample = rSample
    fGrid,rGrid = np.meshgrid(fSample,newRSample,indexing="ij")
    fluxGrid = rGrid**2*emmInt(newRSample,fSample)/(dl**2 + rGrid**2)
    return integrate(fluxGrid,rGrid)*3.09e24*1.60e20*boostMod

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