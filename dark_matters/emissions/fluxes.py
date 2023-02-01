import numpy as np 
from scipy.integrate import simps as integrate
from scipy.interpolate import interp1d,interp2d
from astropy import constants as const

def surfaceBrightnessLoop(nu_sb,fSample,rSample,emm,deltaOmega=4*np.pi):
    """
    Surface brightness from emmissivity 

    Arguments
    ---------------------------
    nu_sb : float
        Output frequency [MHz]
    fSample : array-like float (m)
        All sampled frequencies [Mpc]
    rSample : array-like float (n)
        Sampled radii for emissivity [Mpc]
    emm : array-like float (m,n)
        Emissivity [GeV cm^-3]
    deltaOmega : float
        Angular area photons are distributed over [sr]

    Returns
    ---------------------------
    surface_brightness : array-like float (n)
        Surface brightness at frequency nu_sb [Jy arcminute^-2]
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
    Flux calculated from an emisssivity

    Arguments
    ---------------------------
    rf : float
        Radial integration limit [Mpc]
    dl : float
        Luminosity distance [Mpc] 
    fSample : array-like float (n)
        Output frequency values [MHz]
    rSample : array-like float (m)
        Sampled radii for emissivity [Mpc]
    emm : array-like float (n,m) 
        Emissivity [GeV cm^-3]
    boostMod : float
        Flux boosting factor

    Returns
    ---------------------------
    flux : array-like float (n)
        Output fluxes [Jy]
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
    Prompt gamma-ray flux determined via J-factor

    Arguments
    ---------------------------
    mx : float
        WIMP mass [GeV]
    z : float
        Halo redshift
    jFactor : float
        The "J-factor" of the halo [GeV^2/cm^5]
    fSample : array-like float
        Output frequency values [MHz]
    gSample : array-like float
        Yield spectrum Lorentz-gamma values
    qSample : array-like float
        (Yield spectrum * electron mass) [particles per annihilation]
    mode_exp : float
        2 for annihilation, 1 for decay

    Returns
    ---------------------------
    flux : array-like float
        Output flux from annihilation in GeV cm^-2 s^-1
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
            emm[i] = Q_func(E_g)*jFactor*nwimp0*E_g #units of cm^-2 s^-1
            #print halo.J,Q_func(E_g)*nwimp0*E_g/(1+halo.z),1+halo.z
    return 2.0*emm*const.h.to("J/Hz").value/1e-26*1e4 #2 gamma-rays per event in Jy (cm^-2 -> m^-2 (1e4), Jy = 1e-26 W/m^2/Hz)