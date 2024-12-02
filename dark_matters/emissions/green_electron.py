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
def read_electrons_c(infile,e_set,r_sample):
    """
    Read the output from the c executable that finds equilibrium electon distributions

    Arguments
    ---------------------------
    infile: str
        File path to c output
    e_set : array-like float (n)
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
    e_array = np.array(line,dtype=float)
    n = len(r_sample)
    k = len(e_set)
    electrons = np.zeros((k,n),dtype=float)
    for i in range(0,k):
        for j in range(0,n):
            electrons[i][j] = e_array[i*n + j]
    inf.close()
    return electrons

#write input file for c executable
def write_electrons_c(outfile,k_prime,e_set,q_set,ngr,r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,z,delta,diff,d0,u_ph,num_threads,num_images):
    """
    Write the input file for the c executable that finds equilibrium electon distributions

    Arguments
    ---------------------------
    outfile : str
        Path to output file from C++
    k_prime : int
        Number of energy samples for integration
    e_set : array-like float (k)
        Yield function Lrentz-gamma values
    q_set : array-like float (k)
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
    u_ph : float
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
    outf.write(f"{len(e_set)} {k_prime} {len(r_sample)} {ngr}\n")
    for r in r_sample:
        outf.write(f"{r} ")
    outf.write("\n")
    for x in e_set:
        outf.write(f"{x} ")
    outf.write("\n")
    for x in q_set:
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
    outf.write(f"{diff} {u_ph} {d0} {mode_exp} {num_threads:d} {num_images:d}")
    outf.close()

#run the c executable with a written infile and retrieve output
def electrons_from_c(outfile,infile,exec_electron_c,k_prime,e_set,q_set,ngr,r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,z,delta,diff,d0,u_ph,num_threads=1,num_images=51):
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
    k_prime : int
        Number of energy samples for integration
    e_set : array-like float (k)
        Yield function Lrentz-gamma values
    q_set : array-like float (k)
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
    u_ph : float
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
    write_electrons_c(outfile,k_prime,e_set,q_set,ngr,r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,z,delta,diff,d0,u_ph,num_threads,num_images)
    if not os.path.isfile(exec_electron_c):
        return None
    try:
        if platform.system() == "Linux" or platform.system() == "Darwin":
            call([exec_electron_c+" "+outfile+" "+infile],shell=True)#,cwd=cdir)
        else:
            call([exec_electron_c,outfile,infile],shell=True)#,cwd=cdir)
    except:
        return None
    electron_data = read_electrons_c(infile,e_set,r_sample)
    return electron_data

def eloss_vector(E_vec,B,ne,z,u_ph=0.0):
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
    u_ph : float
        Ambient photon energy density [eV cm^-3]

    Returns
    ---------------------------
    eloss_vector : array-like float
        Loss function at E_vec [GeV s^-1]
    """
    me = (constants.m_e*constants.c**2).to("GeV").value 
    coeffs = np.array([0.76e-16*u_ph+0.25e-16*(1+z)**4,0.0254e-16,6.13e-16,4.7e-16],dtype=float)
    eloss_tot = coeffs[0]*(me*E_vec)**2 + coeffs[1]*(me*E_vec)**2*B**2 + coeffs[2]*ne*(1+np.log(E_vec/ne)/75.0)+ coeffs[3]*ne*E_vec*me
    return eloss_tot/me #make it gamma s^-1 units

def diff_func_normed(gamma,delta):
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
    diff_func : array-like float
        Diffusion function normalised
    """
    me = (constants.m_e*constants.c**2).to("GeV").value
    E = gamma*me
    return E**(delta)

def v_func(mx,gamma,B,ne,delta,z,u_ph):
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
    u_ph : float
        Ambient photon energy density [eV cm^-3]
    
    Returns
    ---------------------------
    v_func : array-like float
        V function [Gev^-1 s]
    """
    me = (constants.m_e*constants.c**2).to("GeV").value
    gamma_prime = np.logspace(np.log10(gamma),np.log10(mx/me*np.ones_like(gamma)),num=101,axis=-1) 
    return integrate(y=diff_func_normed(gamma_prime,delta)/eloss_vector(gamma_prime,B,ne,z,u_ph),x=gamma_prime)

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

def equilibrium_electrons_grid_partial(k_prime,e_set,q_set,n_prime,r_sample,rho_dm_sample,b_set,ne_set,mx,mode_exp,b_av,ne_av,z,delta,diff,d0,u_ph,num_threads,num_images):
    """
    Calculates equilibrium electron distribution via Green's function

    Arguments
    ---------------------------
    k_prime : int
        Number of energy samples for integration
    e_set : array-like float (k)
        Yield function Lrentz-gamma values
    q_set : array-like float (k)
        (Yield function * electron mass) [particles per annihilation]
    n_prime : int
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
    u_ph : float
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

    k = len(e_set) #number of energy bins
    n = len(r_sample)
    
    electrons = np.zeros((k,n))
    unit_factor = (1*units.Unit("Msun/Mpc^3")*constants.c**2).to("GeV/cm^3").value
    nwimp0 = unit_factor**mode_exp/mode_exp/mx**mode_exp  #non-thermal wimp density (cm^-3) (central)
    rhodm = nwimp0*rho_dm_sample**mode_exp
    rho_intp = interp1d(r_sample,rhodm)
    images = np.arange(-(num_images),(num_images+1),dtype=int)

    with np.errstate(invalid="ignore",divide="ignore"):
        v_sample = v_func(mx,e_set,b_av,ne_av,delta,z,u_ph)*d0/3.086e24**2
    v_sample = np.where(np.isnan(v_sample),0.0,v_sample)
    v_intp = interp1d(e_set,v_sample)

    me = (constants.m_e*constants.c**2).to("GeV").value
    def kernel(i):
        E = e_set[i]
        vE = v_intp(e_set[i])
        e_prime = np.logspace(np.log10(E),np.log10(mx/me),num=k_prime)
        v_e_prime = v_intp(e_prime)
        q_grid = interp1d(e_set,q_set)(e_prime)
        delta_v = np.tensordot(np.ones(k_prime),vE,axes=0) - v_e_prime
        delta_v = np.where(delta_v<0,0.0,delta_v)
        for j in np.arange(0,n-1):
            if diff == 1:
                delta_v_grid = np.tensordot(delta_v,np.ones_like(images),axes=0)
                image_grid = np.tensordot(np.ones_like(delta_v),images,axes=0)
                r_n_grid = (-1.0)**image_grid*r_sample[j] + 2*image_grid*r_sample[-1]
                
                r_central = np.abs(r_n_grid)
                r_central = np.where(r_central > r_sample[-1],r_sample[-1],r_central)
                r_central = np.where(r_central < r_sample[0],r_sample[0],r_central)
                r_max = r_central + np.sqrt(delta_v_grid)*10
                r_min = r_central - np.sqrt(delta_v_grid)*10
                r_min = np.where(r_min < r_sample[0],r_sample[0],r_min)
                r_max = np.where(r_max > r_sample[-1],r_sample[-1],r_max)
                r_prime_grid = np.linspace(r_min,r_max,num=n_prime,axis=-1)

                image_grid = np.tensordot(image_grid,np.ones(n_prime),axes=0)
                delta_v_grid = np.tensordot(delta_v_grid,np.ones(n_prime),axes=0)
                r_n_grid = np.tensordot(r_n_grid,np.ones(n_prime),axes=0)
                rho_prime_dm_grid = rho_intp(r_prime_grid)

                with np.errstate(invalid="ignore",divide="ignore",over="ignore"): #ignore issues from exponential, fix below
                    G = r_prime_grid/r_n_grid*(np.exp(-0.25*(r_prime_grid-r_n_grid)**2/delta_v_grid) - np.exp(-0.25*(r_prime_grid+r_n_grid)**2/delta_v_grid))*(-1.0)**image_grid/np.sqrt(4*np.pi*delta_v_grid)
                G *= rho_prime_dm_grid/rhodm[j]
                G = np.where(np.isnan(G),0.0,G)
                with np.errstate(invalid="ignore",divide="ignore",over="ignore"): #ignore cases where r_prime is constant (should integrate to zero anyway)
                    G = integrate(y=G,x=r_prime_grid,axis=-1)
                G = np.sum(G,axis=-1) #now e_prime by eGrid in shape
                G[0] = 1.0
                #G = np.where(delta_v==0.0,1.0,G)
            else:
                G = np.ones_like(e_prime)

            electrons[i,j] = booles_rule_log10(G*q_grid,e_prime,axis=-1)*rhodm[j]/eloss_vector(e_set[i],b_set[j],ne_set[j],z,u_ph) 
    Parallel(n_jobs=num_threads,require='sharedmem')(delayed(kernel)(i) for i in tqdm(range(k)))
    electrons = np.where(np.isnan(electrons),0.0,electrons)
    return electrons