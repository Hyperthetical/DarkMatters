"""
DarkMatters.emissions module for calculating electron equilibrium distributions with Green's functions
"""
import numpy as np
import platform
from scipy.integrate import simpson as integrate
from scipy.interpolate import interp1d
from .progress_bar import progress
from subprocess import call
from astropy import constants,units
import os
from joblib import Parallel, delayed
from tqdm import tqdm


#read the output from the c routine
def read_electrons_c(infile,E_set,r_sample):
    """
    Read the output from the c executable that finds equilibrium electon distributions

    Arguments
    ---------------------------
    infile: str
        File path to c output
    E_set : array-like float (n)
        Electron energy samples
    r_sample : array-like float (m)
        Radial samples

    Returns
    ---------------------------
    electrons : array-like float (n,m)
        Electron equilibrium distributions
    """
    try:
        inf = open(infile,"r")
    except:
        return None
    line = inf.readline().strip().split()
    eArray = np.array(line,dtype=float)
    n = len(r_sample)
    k = len(E_set)
    electrons = np.zeros((k,n),dtype=float)
    for i in range(0,k):
        for j in range(0,n):
            electrons[i][j] = eArray[i*n + j]
    inf.close()
    return electrons

#write input file for c executable
def write_electrons_c(outfile,kPrime,E_set,Q_set,ngr,r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,z,delta,diff,d0,uPh,num_threads,num_images):
    """
    Write the input file for the c executable that finds equilibrium electon distributions

    Arguments
    ---------------------------
    outfile : str
        Path to output file from C++
    kPrime : int
        Number of energy samples for integration
    E_set : array-like float (k)
        Yield function Lrentz-gamma values
    Q_set : array-like float (k)
        (Yield function * electron mass) [particles per annihilation]
    ngr : int
        Number of radial samples for integration
    r_sample : array-like float (n)
        Sampled radii [Mpc]
    rho_dm_sample : array-like float (n)
        Dark matter density at r_sample [Msun/Mpc^3]
    b_set : array-like float (n)
        Magnetic field strength at r_sample [uG]
    ne_set : array-like float (n)
        Gas density at r_sample [cm^-3]
    mx : float
        WIMP mass [GeV]
    mode_exp : float
        2 for annihilation, 1 for decay
    b_av : float
        Magnetic field spatial average [uG]
    ne_av : float
        Gas density spatial average [cm^-3]
    z : float
        Redshift of halo
    delta : float
        Difusion power-spectrum index
    diff : int
        1 for difusion, 0 for loss-only
    uPh : float
        Ambient photon energy density [eV cm^-3]
    num_threads : int
        Number of threads for parallel processing
    num_images : int
        Number of image charges for Green's solution

    Returns
    -------------------------
    None
    """
    outf = open(outfile,"w")
    outf.write(f"{len(E_set)} {kPrime} {len(r_sample)} {ngr}\n")
    for r in r_sample:
        outf.write(f"{r} ")
    outf.write("\n")
    for x in E_set:
        outf.write(f"{x} ")
    outf.write("\n")
    for x in Q_set:
        outf.write(f"{x} ")
    outf.write("\n")
    for r in rho_dm_sample:
        outf.write(f"{r} ")
    outf.write("\n")
    for x in b_sample:
        outf.write(f"{x} ")
    outf.write("\n")
    for x in ne_sample:
        outf.write(f"{x} ")
    outf.write("\n")
    outf.write(f"{z} {mx} {delta} {b_av} {ne_av}\n")
    outf.write(f"{diff} {uPh} {d0} {mode_exp} {num_threads:d} {num_images:d}")
    outf.close()

#run the c executable with a written infile and retrieve output
def electrons_from_c(outfile,infile,exec_electron_c,kPrime,E_set,Q_set,ngr,r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,z,delta,diff,d0,uPh,num_threads=1,num_images=51):
    """
    Prepare the input file, run the c executable that finds equilibrium electon distributions and retrieve the output

    Arguments
    ---------------------------
    outfile : str
        Path to output file from C++
    infile : str
        Path to input file for C++
    exec_electron_c : str
        Path to C++ executable
    kPrime : int
        Number of energy samples for integration
    E_set : array-like float (k)
        Yield function Lrentz-gamma values
    Q_set : array-like float (k)
        (Yield function * electron mass) [particles per annihilation]
    ngr : int
        Number of radial samples for integration
    r_sample : array-like float (n)
        Sampled radii [Mpc]
    rho_dm_sample : array-like float (n)
        Dark matter density at r_sample [Msun/Mpc^3]
    b_set : array-like float (n)
        Magnetic field strength at r_sample [uG]
    ne_set : array-like float (n)
        Gas density at r_sample [cm^-3]
    mx : float
        WIMP mass [GeV]
    mode_exp : float
        2 for annihilation, 1 for decay
    b_av : float
        Magnetic field spatial average [uG]
    ne_av : float
        Gas density spatial average [cm^-3]
    z : float
        Redshift of halo
    delta : float
        Difusion power-spectrum index
    diff : int
        1 for difusion, 0 for loss-only
    uPh : float
        Ambient photon energy density [eV cm^-3]
    num_threads : int
        Number of threads for parallel processing
    num_images : int
        Number of image charges for Green's solution

    Returns
    ---------------------------
    electrons : array-like float (k,n)
        Electron equilibrium distributions [GeV cm^-3]
    """
    write_electrons_c(outfile,kPrime,E_set,Q_set,ngr,r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,z,delta,diff,d0,uPh,num_threads,num_images)
    if not os.path.isfile(exec_electron_c):
        return None
    try:
        if platform.system() == "Linux" or platform.system() == "Darwin":
            call([exec_electron_c+" "+outfile+" "+infile],shell=True)#,cwd=cdir)
        else:
            call([exec_electron_c,outfile,infile],shell=True)#,cwd=cdir)
    except:
        return None
    electronData = read_electrons_c(infile,E_set,r_sample)
    return electronData

def eloss_vector(E_vec,B,ne,z,uPh=0.0):
    """
    Calculates a vectorised form of the energy loss function b(E)

    Arguments
    ---------------------------
    E_vec : array-like float
        Electron Lorentz-gamma factors
    B : float
        Magnetic field strength [uG]
    ne : float
        Gas density [cm^-3]
    z : float
        Halo redshift
    uPh : float
        Ambient photon energy density [eV cm^-3]

    Returns
    ---------------------------
    eloss_vector : array-like float
        Loss function at E_vec [GeV s^-1]
    """
    me = (constants.m_e*constants.c**2).to("GeV").value 
    coeffs = np.array([0.76e-16*uPh+0.25e-16*(1+z)**4,0.0254e-16,6.13e-16,4.7e-16],dtype=float)
    eloss_tot = coeffs[0]*(me*E_vec)**2 + coeffs[1]*(me*E_vec)**2*B**2 + coeffs[2]*ne*(1+np.log(E_vec/ne)/75.0)+ coeffs[3]*ne*E_vec*me
    return eloss_tot/me #make it gamma s^-1 units

def diffFuncNormed(gamma,delta):
    """
    Normalised diffusion function

    Arguments
    ---------------------------
    gamma : array-like float
        electron Lorentz-gamma 
    delta : float
        Diffusion power-spectrum index
    
    Returns
    ---------------------------
    diffFunc : array-like float
        Diffusion function normalised
    """
    me = (constants.m_e*constants.c**2).to("GeV").value
    E = gamma*me
    return E**(delta)

def vFunc(mx,gamma,B,ne,delta,z,uPh):
    """
    V function

    Arguments
    ---------------------------
    mx : float
        WIMP mass [GeV]
    gamma : array-like float
        electron Lorentz-gamma 
    B : float
        Average magnetic field [uG]
    ne : float
        Average gas density [cm^-3]
    delta : float
        Diffusion power-spectrum index
    z : float
        Halo redshift
    uPh : float
        Ambient photon energy density [eV cm^-3]
    
    Returns
    ---------------------------
    vFunc : array-like float
        V function [Gev^-1 s]
    """
    me = (constants.m_e*constants.c**2).to("GeV").value
    gammaPrime = np.logspace(np.log10(gamma),np.log10(mx/me*np.ones_like(gamma)),num=101,axis=-1) 
    return integrate(diffFuncNormed(gammaPrime,delta)/eloss_vector(gammaPrime,B,ne,z,uPh),gammaPrime)

def booles_rule_lin(y, x, axis=-1):
    """
    Boole's rule for linearly-spaced data

    Arguments
    ---------------------------
    y : array-like float (n)
        Integrand values
    x : array-like float (n)
        Integrand abscissa
    axis : int
        Axis to be integrated over

    Returns
    ---------------------------
    Integral : array-like float (n)
        Result of integration
    """
    def tupleset(t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)
    y = np.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    returnshape = 0
    start = 0
    stop = N-4
    step = 4
    if x is not None:
        x = np.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))

    slice_all = (slice(None),)*nd
    slice0 = tupleset(slice_all, axis, slice(start,stop,step))
    slice1 = tupleset(slice_all, axis, slice(start+1,stop+1,step))
    slice2 = tupleset(slice_all, axis, slice(start+2,stop+2,step))
    slice3 = tupleset(slice_all, axis, slice(start+3,stop+3,step))
    slice4 = tupleset(slice_all, axis, slice(start+4,stop+4,step))

    if len(x.shape) == 1:
        dx = (x[-1]-x[0])/(N-1)
    else:
        dx = (x[axis].flatten()[-1]-x[axis].flatten()[0])/(N-1)

    result = np.sum(y[slice0]*7 + y[slice1]*32 + y[slice2]*12 + y[slice3]*32 + y[slice4]*7,axis=axis)*dx*2/45

    if returnshape:
        x = x.reshape(saveshape)

    return result

def booles_rule_log10(y, x, axis=-1):
    """
    Boole's rule for log-spaced data

    Arguments
    ---------------------------
    y : array-like float (n)
        Integrand values
    x : array-like float (n)
        Integrand abscissa
    axis : int
        Axis to be integrated over

    Returns
    ---------------------------
    Integral : array-like float (n)
        Result of integration
    """
    def tupleset(t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)
    y = np.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    returnshape = 0
    start = 0
    stop = N-4
    step = 4
    if x is not None:
        x = np.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))

    slice_all = (slice(None),)*nd
    slice0 = tupleset(slice_all, axis, slice(start,stop,step))
    slice1 = tupleset(slice_all, axis, slice(start+1,stop+1,step))
    slice2 = tupleset(slice_all, axis, slice(start+2,stop+2,step))
    slice3 = tupleset(slice_all, axis, slice(start+3,stop+3,step))
    slice4 = tupleset(slice_all, axis, slice(start+4,stop+4,step))

    if len(x.shape) == 1:
        dx = (np.log10(x[-1])-np.log10(x[0]))/(N-1)
    else:
        dx = (np.log10(x[axis][-1])-np.log10(x[axis][0]))/(N-1)

    result = np.sum(y[slice0]*x[slice0]*7 + y[slice1]*x[slice1]*32 + y[slice2]*x[slice2]*12 + y[slice3]*x[slice3]*32 + y[slice4]*x[slice4]*7,axis=axis)*dx*2/45*np.log(10.0)

    if returnshape:
        x = x.reshape(saveshape)

    return result

def equilibriumElectronsGridPartial(kPrime,E_set,Q_set,nPrime,r_sample,rho_dm_sample,b_set,ne_set,mx,mode_exp,b_av,ne_av,z,delta,diff,d0,uPh,num_threads,num_images):
    """
    Calculates equilibrium electron distribution via Green's function

    Arguments
    ---------------------------
    kPrime : int
        Number of energy samples for integration
    E_set : array-like float (k)
        Yield function Lrentz-gamma values
    Q_set : array-like float (k)
        (Yield function * electron mass) [particles per annihilation]
    nPrime : int
        Number of radial samples for integration
    r_sample : array-like float (n)
        Sampled radii [Mpc]
    rho_dm_sample : array-like float (n)
        Dark matter density at r_sample [Msun/Mpc^3]
    b_set : array-like float (n)
        Magnetic field strength at r_sample [uG]
    ne_set : array-like float (n)
        Gas density at r_sample [cm^-3]
    mx : float
        WIMP mass [GeV]
    mode_exp : float
        2 for annihilation, 1 for decay
    b_av : float
        Magnetic field spatial average [uG]
    ne_av : float
        Gas density spatial average [cm^-3]
    z : float
        Redshift of halo
    delta : float
        Difusion power-spectrum index
    diff : int
        1 for difusion, 0 for loss-only
    uPh : float
        Ambient photon energy density [eV cm^-3]
    num_threads : int
        Number of threads for parallel processing
    num_images : int
        Number of image charges for Green's solution

    Returns
    ---------------------------
    electrons : array-like float (k,n)
        Electron distribution / cross-section [GeV cm^-6 s]
    """

    k = len(E_set) #number of energy bins
    n = len(r_sample)
    
    electrons = np.zeros((k,n))
    unit_factor = (1*units.Unit("Msun/Mpc^3")*constants.c**2).to("GeV/cm^3").value
    nwimp0 = unit_factor**mode_exp/mode_exp/mx**mode_exp  #non-thermal wimp density (cm^-3) (central)
    rhodm = nwimp0*rho_dm_sample**mode_exp
    rhoIntp = interp1d(r_sample,rhodm)
    images = np.arange(-(num_images),(num_images+1),dtype=int)

    with np.errstate(invalid="ignore",divide="ignore"):
        vSample = vFunc(mx,E_set,b_av,ne_av,delta,z,uPh)*d0/3.086e24**2
    vSample = np.where(np.isnan(vSample),0.0,vSample)
    vIntp = interp1d(E_set,vSample)

    me = (constants.m_e*constants.c**2).to("GeV").value
    def kernel(i):
        E = E_set[i]
        vE = vIntp(E_set[i])
        ePrime = np.logspace(np.log10(E),np.log10(mx/me),num=kPrime)
        vEPrime = vIntp(ePrime)
        qGrid = interp1d(E_set,Q_set)(ePrime)
        deltaV = np.tensordot(np.ones(kPrime),vE,axes=0) - vEPrime
        deltaV = np.where(deltaV<0,0.0,deltaV)
        for j in np.arange(0,n-1):
            if diff == 1:
                deltaVGrid = np.tensordot(deltaV,np.ones_like(images),axes=0)
                imageGrid = np.tensordot(np.ones_like(deltaV),images,axes=0)
                rNGrid = (-1.0)**imageGrid*r_sample[j] + 2*imageGrid*r_sample[-1]
                
                rCentral = np.abs(rNGrid)
                rCentral = np.where(rCentral > r_sample[-1],r_sample[-1],rCentral)
                rCentral = np.where(rCentral < r_sample[0],r_sample[0],rCentral)
                rMax = rCentral + np.sqrt(deltaVGrid)*10
                rMin = rCentral - np.sqrt(deltaVGrid)*10
                rMin = np.where(rMin < r_sample[0],r_sample[0],rMin)
                rMax = np.where(rMax > r_sample[-1],r_sample[-1],rMax)
                rPrimeGrid = np.linspace(rMin,rMax,num=nPrime,axis=-1)

                imageGrid = np.tensordot(imageGrid,np.ones(nPrime),axes=0)
                deltaVGrid = np.tensordot(deltaVGrid,np.ones(nPrime),axes=0)
                rNGrid = np.tensordot(rNGrid,np.ones(nPrime),axes=0)
                rhoPrimeDMGrid = rhoIntp(rPrimeGrid)

                with np.errstate(invalid="ignore",divide="ignore",over="ignore"): #ignore issues from exponential, fix below
                    G = rPrimeGrid/rNGrid*(np.exp(-0.25*(rPrimeGrid-rNGrid)**2/deltaVGrid) - np.exp(-0.25*(rPrimeGrid+rNGrid)**2/deltaVGrid))*(-1.0)**imageGrid/np.sqrt(4*np.pi*deltaVGrid)
                G *= rhoPrimeDMGrid/rhodm[j]
                G = np.where(np.isnan(G),0.0,G)
                with np.errstate(invalid="ignore",divide="ignore",over="ignore"): #ignore cases where rPrime is constant (should integrate to zero anyway)
                    G = integrate(G,rPrimeGrid,axis=-1)
                G = np.sum(G,axis=-1) #now ePrime by eGrid in shape
                G[0] = 1.0
                #G = np.where(deltaV==0.0,1.0,G)
            else:
                G = np.ones_like(ePrime)

            electrons[i,j] = booles_rule_log10(G*qGrid,ePrime,axis=-1)*rhodm[j]/eloss_vector(E_set[i],b_set[j],ne_set[j],z,uPh) 
    Parallel(n_jobs=num_threads,require='sharedmem')(delayed(kernel)(i) for i in tqdm(range(k)))
    electrons = np.where(np.isnan(electrons),0.0,electrons)
    return electrons