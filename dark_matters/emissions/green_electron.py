import numpy as np
import platform
from scipy.integrate import simps as integrate
from scipy.interpolate import interp1d
from .progress_bar import progress
from subprocess import call
from astropy import constants as const
import os
from joblib import Parallel, delayed
from tqdm import tqdm


#read the output from the c routine
def read_electrons_c(infile,E_set,r_sample):
    """
    Read the output from the c executable that finds equilibrium electon distributions
        ---------------------------
        Parameters
        ---------------------------
        infile   - Required : file path to c output (String)
        halo     - Required : halo envionment (halo_env)
        phys     - Required : physical environment (phys_env)
        sim      - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        2D array of floats, electron equilibrium distributions (phys.e_bins x sim.n)
    """
    try:
        inf = open(infile,"r")
    except:
        return None
    line = inf.readline().strip().split()
    eArray = np.array(line,dtype=float)
    n = len(r_sample[0])
    k = len(E_set)
    electrons = np.zeros((k,n),dtype=float)
    for i in range(0,k):
        for j in range(0,n):
            electrons[i][j] = eArray[i*n + j]
    inf.close()
    return electrons

#write input file for c executable
def write_electrons_c(outfile,kPrime,E_set,Q_set,r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,z,lc,delta,diff,d0,ISRF,num_threads,num_images):
    """
    Write the input file for the c executable that finds equilibrium electon distributions
        ---------------------------
        Parameters
        ---------------------------
        outfile  - Required : file inputs to c executable go to (String)
        halo     - Required : halo envionment (halo_env)
        phys     - Required : physical environment (phys_env)
        sim      - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        None
    """
    outf = open(outfile,"w")
    outf.write(str(len(E_set))+" "+str(kPrime)+" "+str(len(r_sample[0]))+" "+str(len(r_sample[1]))+"\n")
    for r in r_sample[0]:
        outf.write(str(r)+" ")
    outf.write("\n")
    for r in r_sample[1]:
        outf.write(str(r)+" ")
    outf.write("\n")
    for x in E_set:
        outf.write(str(x)+" ")
    outf.write("\n")
    for x in Q_set:
        outf.write(str(x)+" ")
    outf.write("\n")
    for r in rho_dm_sample[0]:
        outf.write(str(r)+" ")
    outf.write("\n")
    for r in rho_dm_sample[1]:
        outf.write(str(r)+" ")
    outf.write("\n")
    for x in b_sample:
        outf.write(str(x)+" ")
    outf.write("\n")
    for x in ne_sample:
        outf.write(str(x)+" ")
    outf.write("\n")
    outf.write(str(z)+" "+str(mx)+" "+str(lc)+" "+str(delta)+" "+str(b_av)+" "+str(ne_av)+"\n")
    outf.write(f"{diff} {ISRF:d} {d0} {mode_exp} {num_threads:d} {num_images:d}")
    outf.close()

#run the c executable with a written infile and retrieve output
def electrons_from_c(outfile,infile,exec_electron_c,kPrime,E_set,Q_set,r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,z,lc,delta,diff,d0,ISRF,num_threads=1,num_images=51):
    """
    Prepare the input file, run the c executable that finds equilibrium electon distributions and retrieve the output
        ---------------------------
        Parameters
        ---------------------------
        outfile  - Required : file inputs to c executable go to (String)
        infile   - Required : file path to c output (String)
        halo     - Required : halo envionment (halo_env)
        phys     - Required : physical environment (phys_env)
        sim      - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        2D array of floats, electron equilibrium distributions (phys.e_bins x sim.n)
    """
    write_electrons_c(outfile,kPrime,E_set,Q_set,r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,z,lc,delta,diff,d0,ISRF,num_threads,num_images)
    if not os.path.isfile(exec_electron_c):
        return None
    try:
        if platform.system() == "Linux":
            call([exec_electron_c+" "+outfile+" "+infile],shell=True)#,cwd=cdir)
        else:
            call([exec_electron_c,outfile,infile],shell=True)#,cwd=cdir)
    except:
        return None
    electronData = read_electrons_c(infile,E_set,r_sample)
    return electronData

#@jit
def eloss_vector(E_vec,B,ne,z,ISRF=0):
    """
    Calculates a vectorised form of the energy loss function b(E)
        ---------------------------
        Parameters
        ---------------------------
        E_vec  - Required : set of energy values for b(E) (float)
        B      - Required : average magnetic field strength (float)
        ne     - Required : average gas density (float)
        z      - Required : redshift of the halo bing considered (float)
        ISRF   - Optional : flag whether to include ISRF in inverse-compton losses (int, 1 or 0)
        ---------------------------
        Output
        ---------------------------
        1D float array (sim.n)
    """
    me = (const.m_e*const.c**2).to("GeV").value 
    coeffs = np.array([6.08e-16+0.25e-16*(1+z)**4,0.0254e-16,6.13e-16,4.7e-16],dtype=float)
    if ISRF == 0:
        coeffs[0] = 0.25e-16*(1+z)**4 #only CMB used so it scales with z
    eloss_tot = coeffs[0]*(me*E_vec)**2 + coeffs[1]*(me*E_vec)**2*B**2 + coeffs[2]*ne*(1+np.log(E_vec/ne)/75.0)+ coeffs[3]*ne*E_vec*me
    return eloss_tot/me #make it gamma s^-1 units

def diffFuncNormed(gamma,B,lc,delta):
    me = (const.m_e*const.c**2).to("GeV").value
    E = gamma*me
    return E**(2-delta)*B**(delta-2)*lc**(delta-1)

def vFunc(mx,gamma,B,ne,lc,delta,z,ISRF):
    me = (const.m_e*const.c**2).to("GeV").value
    gammaPrime = np.logspace(np.log10(gamma),np.log10(mx/me*np.ones_like(gamma)),num=100,axis=-1) 
    return integrate(diffFuncNormed(gammaPrime,B,lc,delta)/eloss_vector(gammaPrime,B,ne,z,ISRF),gammaPrime)

##@jit
def green_integrand_vector(rpr,rn,dv,rhosq):
    """
    Calculates and returns the integrand of the Green's function for electron diffusion
        ---------------------------
        Parameters
        ---------------------------
        rpr   - Required : set of r points in the integrand of the Green's function (float)
        rn    - Required : bounday condition poisiton of image charge (float)
        dv    - Required : diffusion scale parameter (float)
        rhosq - Required : ratio of rho^2 at r and each rpr point (float)
        ---------------------------
        Output
        ---------------------------
        1D array of floats (len(rpr)) 
    """
    return rpr/rn*(np.exp(-(rpr-rn)**2/(4.0*dv))-np.exp(-(rpr+rn)**2/(4.0*dv)))*rhosq

#@jit
def Green_vector(r_set,r,rhosq,dv,diff):
    """
    Calculates and returns a vector of Green's function values
        ---------------------------
        Parameters
        ---------------------------
        r_set - Required : set of all r points in Green integrand (float)
        r     - Required : position to calculate at (float)
        dv    - Required : diffusion scale parameter (float)
        rhosq - Required : ratio of rho^2 at r and each r_set point (float array)
        diff  - Required : diffusion flag (0 or 1)
        ---------------------------
        Output
        ---------------------------
        1D float array (phys.e_bins)
    """
    m = len(dv)
    G = np.zeros(m)
    if(diff == 0):
        G[dv != -1] = 1.0
    else:
        n = len(r_set)
        rh = r_set.max()
        images = 33 #image charges for green function solution -> should be odd
        image_set = np.arange(-images,images+1,dtype=int)

        for i in range(0,m):
            if(dv[i] == 0.0):
                G[i] = 1.0 #this is a no diffusion case or E = Epr case
            elif(dv[i] == -1): #this exists to limit the domain of the Green's function to between relevant energies
                G[i] = 0.0
            else:
                k1 = 1.0/np.sqrt(4*np.pi*dv[i]) #coefficient
                for j in range(0,len(image_set)):
                    p = image_set[j]
                    rn = (-1.0)**p*r + 2.0*p*rh  #position of the image charge
                    print(k1,dv[i],Green_p(r,rn,r_set,rhosq,dv[i],p))
                    G[i] = G[i] + k1*Green_p(r,rn,r_set,rhosq,dv[i],p)
    return G

#@jit
def Green_p(r,rn,r_set,rhosq,dv,p):
    """
    Calculates Green function contribution from an image charge p
        ---------------------------
        Parameters
        ---------------------------
        r     - Required : position to calculate at (float)
        r_set - Required : set of r points in the integrand of the Green's function (float)
        rn    - Required : bounday condition poisiton of image charge (float)
        dv    - Required : diffusion scale parameter (float)
        rhosq - Required : ratio of rho^2 at r and each rpr point (float)
        p     - Required : image charge p
        ---------------------------
        Output
        ---------------------------
        float
    """
    r_int = green_integrand_vector(r_set,rn,dv,rhosq)
    return (-1.0)**p*integrate(r_int,r_set)

def diffusion_constant(halo,phys):
    """
    This sets the phys.d0 value for the diffusion constant
        ---------------------------
        Parameters
        ---------------------------
        halo - Required : DM halo environment (halo_env)
        phys - Required : Physical environment (phys_env)
        ---------------------------
        Output
        ---------------------------
        None - assigned to phys.d0 [cm^2 s^-1]
    """
    d0 = (1.0/3.0)*(3e8)**(phys.delta-1.0)*(1.6e-19)**(phys.delta-2.0)*(1.6e-10)**(2.0-phys.delta)
    d0 = d0*(halo.bav*1e-10)**(phys.delta-2.0)*(phys.lc*3.09e19)**(phys.delta-1.0)*1e4
    phys.d0 = d0 

def equilibrium_p(halo,phys):
    """
    This is a wrapper function to fit into the API of my python code
        ---------------------------
        Parameters
        ---------------------------
        halo - Required : DM halo environment (halo_env)
        phys - Required : Physical environment (phys_env)
        ---------------------------
        Output
        ---------------------------
        2D float array (phys.e_bins x sim.n) 
    """
    return equilibrium_electrons(phys.spectrum[0],phys.spectrum[1],halo.r_sample,halo.rho_dm_sample,phys.mx,halo.mode_exp,halo.bav,halo.neav,halo.z,phys.lc,phys.delta,phys.diff,phys.d0,phys.ISRF)
    #return getElectrons_numeric(halo,phys)

def diffFunc(E_set,B,lc,delta):
    """
    Returns the diffusion function
        ---------------------------
        Parameters
        ---------------------------
        E_set - Required :energy domain
        B     - Required : mean magnetic field
        lc    - Required : minimum homogenuity scale for the field
        delta - Required : turbulence spectral index
        ---------------------------
        Output
        ---------------------------
        1D float array (phys.e_bins)
    """
    me = (const.m_e*const.c**2).to("GeV").value
    d0= (1.0/3.0)*(3e8)**(delta-1.0)*(1.6e-19)**(delta-2.0)*(1.6e-10)**(2.0-delta)
    d0 = d0*(B*1e-10)**(delta-2.0)*(lc*3.09e19)**(delta-1.0)*1e4   #cm^2 s^-1
    d0 = np.exp(np.log(d0)-2.0*np.log(3.09e24))   #Mpc^2 s^-1
    dset = d0*(E_set*me)**(2.0-delta)
    return dset

def equilibriumElectronsGrid(E_set,Q_set,r_sample,rho_dm_sample,mx,mode_exp,b_av,ne_av,z,lc,delta,diff,d0,ISRF):
    """
    Calculates equilibrium electron distribution 2D array (energy)(position)
        ---------------------------
        Parameters
        ---------------------------
        E_set            - Required : an array of E/me, Lorentz gammas
        Q_set            - Required : electron generation function from chi-chi annihilation at each E_set value
        r_sample[0]      - Required : set of radial sampling values 
        r_sample[1]      - Required : set of radial sampling values for the Green's functions
        rho_dm_sample[0] - Required : WIMP pair density at each radial sampling value
        rho_dm_sample[1] - Required : WIMP pair density at each radial sampling value for the Green's functions sampling
        mx               - Required : WIMP mass in GeV
        mode_exp         - Required : 2.0 for annihilation, 1.0 for decay
        b_av             - Required : average magnetic field strength in uG
        ne_av            - Required : average plasma density in cm^-3
        z                - Required : redshift
        lc               - Required : turbulent length scale for B in kpc
        delta            - Required : power-law slope for the B field turbulence spectrum 5/3 Kolmogorov, 2 is Bohm
        diff             - Required : flag, 0 -> no diffusion, 1 -> diffusion
        ISRF             - Required : flag, 0 -> CMB IC only in energy-loss, 1 -> ISRF and CMB
        ---------------------------
        Output
        ---------------------------
        2D float array (phys.e_bins x sim.n)
    """
    k = len(E_set) #number of energy bins
    kPrime = k

    #msun converted to kg, convert to GeV, convert kpc to cm 
    nwimp0 = np.sqrt(1.458e-33)**mode_exp/mode_exp/mx**mode_exp  #non-thermal wimp density (cm^-3) (central)
    rhodm = nwimp0*rho_dm_sample[0]**mode_exp
    rhodm_gr = nwimp0*rho_dm_sample[1]**mode_exp
    

    imageNum = 33
    print(len(np.arange(-(imageNum-1)/2,(imageNum-1)/2,dtype=int)))
    eGrid,rGrid = np.meshgrid(E_set,r_sample[0])
    rGrid = np.tensordot(np.tensordot(np.ones_like(r_sample[1]),np.ones(imageNum-1),axes=0),np.tensordot(np.ones(kPrime),rGrid,axes=0),axes=0)
    rPrimeGrid = np.tensordot(np.tensordot(r_sample[1],np.ones(imageNum-1),axes=0),np.tensordot(np.ones(kPrime),np.ones_like(eGrid),axes=0),axes=0)
    imageGrid = np.tensordot(np.tensordot(np.ones_like(r_sample[1]),np.arange(-(imageNum-1)/2,(imageNum-1)/2,dtype=int),axes=0),np.tensordot(np.ones(kPrime),np.ones_like(eGrid),axes=0),axes=0)
    rNGrid = (-1)**imageGrid*rGrid + 2*imageGrid*r_sample[0][-1]
    rhoDMGrid = np.tensordot(np.tensordot(np.ones_like(r_sample[1]),np.ones(imageNum-1),axes=0),np.tensordot(np.ones(kPrime),np.tensordot(np.ones_like(E_set),rhodm,axes=0),axes=0),axes=0)
    rhoPrimeDMGrid = np.tensordot(np.tensordot(rhodm_gr,np.ones(imageNum-1),axes=0),np.tensordot(np.ones(kPrime),np.ones_like(eGrid),axes=0),axes=0)
    

    ePrime = np.logspace(np.log10(eGrid),np.log10(mx),num=kPrime)
    print(ePrime.shape)
    ePrimePrime = np.logspace(np.log10(ePrime),np.log10(mx),num=kPrime)
    print(ePrimePrime.shape)
    vEPrime = integrate(diffFunc(ePrimePrime,b_av,lc,delta)/eloss_vector(ePrimePrime,b_av,ne_av,z,ISRF),ePrimePrime,axis=0)
    print(vEPrime.shape)
    vE = integrate(diffFunc(ePrime,b_av,lc,delta)/eloss_vector(ePrime,b_av,ne_av,z,ISRF),ePrime,axis=0)
    print(vE.shape)
    deltaV = np.tensordot(np.ones(kPrime),vE,axes=0) - vEPrime
    print(deltaV.shape)
    deltaVGrid = np.tensordot(np.tensordot(np.ones_like(r_sample[1]),np.ones(imageNum-1),axes=0),deltaV,axes=0)
    print(deltaVGrid.shape)
    print(rPrimeGrid.shape)
    print(rNGrid.shape)

    G = rPrimeGrid/rNGrid*(np.exp(-0.25*(rPrimeGrid-rNGrid)**2/deltaVGrid) - np.exp(-0.25*(rPrimeGrid+rNGrid)**2/deltaVGrid))*(-1)**imageGrid/np.sqrt(4*np.pi*deltaVGrid)
    G *= rhoPrimeDMGrid/rhoDMGrid
    G = integrate(G,rPrimeGrid,axis=0)
    G = np.sum(G,axis=0) #now ePrime by eGrid in shape

    electrons = G*interp1d(E_set,Q_set)(ePrime)
    electrons = integrate(electrons,ePrime,axis=0)*2*np.tensordot(np.ones(k),rhodm,axis=0)/eloss_vector(eGrid,b_av,ne_av,z,ISRF)

    return electrons

def booles_rule_log10(y, x, axis):
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

def equilibriumElectronsGridPartial(E_set,Q_set,r_sample,rho_dm_sample,b_set,ne_set,mx,mode_exp,b_av,ne_av,z,lc,delta,diff,d0,ISRF,num_threads,num_images):
    """
    Calculates equilibrium electron distribution 2D array (energy)(position)
        ---------------------------
        Parameters
        ---------------------------
        E_set            - Required : an array of E/me, Lorentz gammas
        Q_set            - Required : electron generation function from chi-chi annihilation at each E_set value
        r_sample[0]      - Required : set of radial sampling values 
        r_sample[1]      - Required : set of radial sampling values for the Green's functions
        rho_dm_sample[0] - Required : WIMP pair density at each radial sampling value
        rho_dm_sample[1] - Required : WIMP pair density at each radial sampling value for the Green's functions sampling
        mx               - Required : WIMP mass in GeV
        mode_exp         - Required : 2.0 for annihilation, 1.0 for decay
        b_av             - Required : average magnetic field strength in uG
        ne_av            - Required : average plasma density in cm^-3
        z                - Required : redshift
        lc               - Required : turbulent length scale for B in kpc
        delta            - Required : power-law slope for the B field turbulence spectrum 5/3 Kolmogorov, 2 is Bohm
        diff             - Required : flag, 0 -> no diffusion, 1 -> diffusion
        ISRF             - Required : flag, 0 -> CMB IC only in energy-loss, 1 -> ISRF and CMB
        ---------------------------
        Output
        ---------------------------
        2D float array (phys.e_bins x sim.n)
    """

    k = len(E_set) #number of energy bins
    kPrime = 200
    n = len(r_sample[0])
    #print(k,n,k*n)
    #msun converted to kg, convert to GeV, convert kpc to cm 
    
    electrons = np.zeros((k,n))
    nwimp0 = np.sqrt(1.458e-33)**mode_exp/mode_exp/mx**mode_exp  #non-thermal wimp density (cm^-3) (central)
    rhodm = nwimp0*rho_dm_sample[0]**mode_exp
    rhodm_gr = nwimp0*rho_dm_sample[1]**mode_exp
    imageNum = num_images
    images = np.arange(-(imageNum),(imageNum+1),dtype=int)
    with np.errstate(invalid="ignore",divide="ignore"):
        vSample = vFunc(mx,E_set,b_av,ne_av,lc,delta,z,ISRF)*d0/3.086e24**2
    vSample = np.where(np.isnan(vSample),0.0,vSample)
    vIntp = interp1d(E_set,vSample)
    me = (const.m_e*const.c**2).to("GeV").value
    def kernel(i):
        E = E_set[i]
        vE = vIntp(E_set[i])
        ePrime = np.logspace(np.log10(E),np.log10(mx/me),num=kPrime)
        vEPrime = vIntp(ePrime)
        qGrid = interp1d(E_set,Q_set)(ePrime)
        deltaV = np.tensordot(np.ones(kPrime),vE,axes=0) - vEPrime
        for j in np.arange(0,n-1):
            if diff == 1:
                rPrimeGrid,imageGrid,deltaVGrid = np.meshgrid(r_sample[1],images,deltaV,indexing="ij")
                rNGrid = (-1.0)**imageGrid*r_sample[0][j] + 2*imageGrid*r_sample[0][-1]
                rhoDMGrid = np.ones_like(imageGrid)*rhodm[j]
                rhoPrimeDMGrid = np.tensordot(np.tensordot(rhodm_gr,np.ones(2*imageNum+1),axes=0),np.ones(kPrime),axes=0)

                with np.errstate(invalid="ignore",divide="ignore",over="ignore"):
                    G = rPrimeGrid/rNGrid*(np.exp(-0.25*(rPrimeGrid-rNGrid)**2/deltaVGrid) - np.exp(-0.25*(rPrimeGrid+rNGrid)**2/deltaVGrid))*(-1.0)**imageGrid/np.sqrt(4*np.pi*deltaVGrid)
                G *= rhoPrimeDMGrid/rhoDMGrid
                G = np.where(np.isnan(G),0.0,G)
                G = integrate(G,rPrimeGrid,axis=0)
                G = np.sum(G,axis=0) #now ePrime by eGrid in shape
                G[0] = 1.0
            else:
                G = np.ones_like(ePrime)
            electrons[i,j] = integrate(G*qGrid,ePrime,axis=0)*rhodm[j]/eloss_vector(E_set[i],b_set[j],ne_set[j],z,ISRF)  #non-thermal wimp density (cm^-3) (central)
    Parallel(n_jobs=num_threads,require='sharedmem')(delayed(kernel)(i) for i in tqdm(range(k)))
    electrons = np.where(np.isnan(electrons),0.0,electrons)
    return electrons


def electronsGridV2(mx,eSample,rSample,qFunc,rhoFunc,bFunc,nFunc,z,lc,b_av,ne_av,delta,diff,d0,ISRF,mode_exp,numImages,nPrime=100,kPrime=100):
    if diff == 1:
        vSample = vFunc(mx,eSample,b_av,ne_av,lc,delta,z,ISRF)
        vIntp = interp1d(eSample,vSample*d0/3.086e24**2)
        rh = rSample[-1]    
        rPrime = np.logspace(np.log10(rSample[0]),np.log10(rh),num=nPrime)
    electrons = np.zeros((len(eSample),len(rSample)))
    for i in np.arange(len(eSample)):
        ePrime = np.logspace(np.log10(eSample[i]),np.log10(mx),num=kPrime)
        if diff == 1:
            images = np.arange(-numImages,numImages+1)
            rGrid,ePrimeGrid,iGrid,rPrimeGrid = np.meshgrid(rSample,ePrime,images,rPrime,indexing="ij")
            print(np.where(np.isnan(rGrid)))
            rNGrid = (-1.0)**iGrid*rGrid + 2*rh*iGrid
            dvGrid = vIntp(eSample[i])-vIntp(ePrimeGrid)
            G = rPrimeGrid/rNGrid*(np.exp(-0.25*(rPrimeGrid-rNGrid)**2/dvGrid) - np.exp(-0.25*(rPrimeGrid+rNGrid)**2/dvGrid))*(-1.0)**iGrid/np.sqrt(4*np.pi*dvGrid)
            G *= rhoFunc(rPrimeGrid)**mode_exp/rhoFunc(rGrid)**mode_exp
            G = np.sum(integrate(G*qFunc(ePrimeGrid),rPrimeGrid),axis=-1)
        else:
            rGrid,ePrimeGrid = np.meshgrid(rSample,ePrime,indexing="ij")
            G = qFunc(ePrimeGrid)
        electrons[i,:] = integrate(G,ePrime,axis=-1)/eloss_vector(eSample[i],bFunc(rSample),nFunc(rSample),z,ISRF)*rhoFunc(rSample)**mode_exp/mode_exp/mx**mode_exp
        progress(i + 1,len(eSample))
    return electrons

        





#@jit(nopython=True,parallel=True)
def equilibrium_electrons(E_set,Q_set,r_sample,rho_dm_sample,mx,mode_exp,b_av,ne_av,z,lc,delta,diff,d0,ISRF):
    """
    Calculates equilibrium electron distribution 2D array (energy)(position)
        ---------------------------
        Parameters
        ---------------------------
        E_set            - Required : an array of E/me, Lorentz gammas
        Q_set            - Required : electron generation function from chi-chi annihilation at each E_set value
        r_sample[0]      - Required : set of radial sampling values 
        r_sample[1]      - Required : set of radial sampling values for the Green's functions
        rho_dm_sample[0] - Required : WIMP pair density at each radial sampling value
        rho_dm_sample[1] - Required : WIMP pair density at each radial sampling value for the Green's functions sampling
        mx               - Required : WIMP mass in GeV
        mode_exp         - Required : 2.0 for annihilation, 1.0 for decay
        b_av             - Required : average magnetic field strength in uG
        ne_av            - Required : average plasma density in cm^-3
        z                - Required : redshift
        lc               - Required : turbulent length scale for B in kpc
        delta            - Required : power-law slope for the B field turbulence spectrum 5/3 Kolmogorov, 2 is Bohm
        diff             - Required : flag, 0 -> no diffusion, 1 -> diffusion
        ISRF             - Required : flag, 0 -> CMB IC only in energy-loss, 1 -> ISRF and CMB
        ---------------------------
        Output
        ---------------------------
        2D float array (phys.e_bins x sim.n)
    """
    k = len(E_set) #number of energy bins
    n = len(r_sample[0])
    ngr = len(r_sample[1])

    loss = np.zeros(k)#,dtype=float)  #energy loss (E,r)
    electrons = np.zeros((k,n))#,dtype=float)  #energy loss (E,r)
    int_E = np.zeros(k)#,dtype=float)  
    int_E2 = np.zeros(k)#,dtype=float)  
    rhosq = np.zeros((n,ngr))#,dtype=float)
    me = (const.m_e*const.c**2).to("GeV").value

    Q_set[Q_set<0.0] = 0.0
    #msun converted to kg, convert to GeV, convert kpc to cm 
    nwimp0 = np.sqrt(1.458e-33)**mode_exp/mode_exp/mx**mode_exp  #non-thermal wimp density (cm^-3) (central)
    rhodm = nwimp0*rho_dm_sample[0]**mode_exp
    rhodm_gr = nwimp0*rho_dm_sample[1]**mode_exp
    loss = eloss_vector(E_set,b_av,ne_av,z,ISRF)
    for i in range(0,n):
        rhosq[i,] = (rhodm_gr/rhodm[i])
    d0 = np.exp(np.log(d0)-2.0*np.log(3.09e24))   #Mpc^2 s^-1 from cm^2 s^-1
    if(diff == 1):
        vtab = make_vtab(E_set,d0,delta,loss,me)
    else:
        vtab = np.zeros(k)
    dv = np.zeros(k)
    for i in range(0,k): #loop over energies
        E = E_set[i]
        if(diff == 1):
            dv = -vtab + vtab[i]
        dv[E_set < E] = -1
        for j in range(0,n):   #loop of r
            r = r_sample[0][j]
            int_E = Q_set*Green_vector(r_sample[1],r,rhosq[j],dv,diff)  #diffusion integrand vectroised over E
            electrons[i][j] = integrate(int_E,E_set)/loss[i]*rhodm[j] #the 2 is for electrons and positrons
            progress(i*n + j + 1,k*n)
    return electrons              

def D(E_set,B,lc,delta):
    """
    Returns the diffusion function
        ---------------------------
        Parameters
        ---------------------------
        E_set - Required :energy domain
        B     - Required : mean magnetic field
        lc    - Required : minimum homogenuity scale for the field
        delta - Required : turbulence spectral index
        ---------------------------
        Output
        ---------------------------
        1D float array (phys.e_bins)
    """
    me = (const.m_e*const.c**2).to("GeV").value
    d0= (1.0/3.0)*(3e8)**(delta-1.0)*(1.6e-19)**(delta-2.0)*(1.6e-10)**(2.0-delta)
    d0 = d0*(B*1e-10)**(delta-2.0)*(lc*3.09e19)**(delta-1.0)*1e4   #cm^2 s^-1
    d0 = np.exp(np.log(d0)-2.0*np.log(3.09e24))   #Mpc^2 s^-1
    dset = d0*(E_set*me)**(2.0-delta)
    return dset

#@jit
def get_v(E,E_set,d0,delta,loss,me): 
    """
    Returns the integrated energy variable for use in the Green's function
        ---------------------------
        Parameters
        ---------------------------
        E     - Required : minimal energy for integration domain, maximal is the wimp mass
        E_set - Required : energy domain
        d0    - Required : diffusion constant in Mpc^2 s^-1
        delta - Required : turbulence spectral index
        loss  - Required : energy loss function at E in GeV s^-1
        me    - Required : electron mass in GeV 
        ---------------------------
        Output
        ---------------------------
        float
    """
    me = (const.m_e*const.c**2).to("GeV").value
    int_v = d0*(E_set*me)**(2.0-delta)/loss
    #int_v[E_set<E] = 0.0 
    v = integrate(int_v,E_set)
    return v

#@jit
def make_vtab(E_set,d0,delta,loss,me):
    """
    Returns a table of integrated energy variables for use in the Green's function
        ---------------------------
        Parameters
        ---------------------------
        E_set - Required : energy domain
        d0    - Required : diffusion constant in Mpc^2 s^-1
        delta - Required : turbulence spectral index
        loss  - Required : energy loss function at E in GeV s^-1
        me    - Required : electron mass in GeV 
        ---------------------------
        Output
        ---------------------------
        1D float array (phys.e_bins)
    """
    k = len(E_set)
    vtab = np.zeros(k)#,dtype=float)
    for i in range(0,k):
        epSet = np.logspace(np.log10(E_set[i]),np.log10(E_set[-1]),num=50)
        vtab[i] = get_v(E_set[i],epSet,d0,delta,loss,me)
    vtab = np.where(np.isnan(vtab),0.0,vtab)
    return vtab

#def print_loss_scales(M,ne,z):
    #this prints out time scales for diffusion and energy loss
    #for halo mass M, thermal plasma density ne and at redshift z
    #it uses the D(E) and energy_loss_vector functions
#    spectrum = tools.read_spectrum("/home/geoff/Coding/Python/wimp/it_data/pos_bb_60GeV.dat")
#    outfile = "/home/geoff/Coding/Python/wimp/bb60_m"+str(int(np.log10(M)))+"_b5_z"+str(z)+"_loss.data"
#    E_set = spectrum[0]
#    Q_set = spectrum[1]
#    loss = eloss_vector(E_set,5.0,ne,z)
#    delta = 5.0/3.0
#    h = 0.673
#    w_m = 0.315
#    rh = cosmology.rvir(M,z,h,w_m) 
#    d0 = D(E_set,5.0,1.0,delta)
#    out = open(outfile,"w")
#    for i in range(0,len(E_set)):
#        out.write(str(E_set[i])+" "+str(rh**2/d0[i])+" "+str(E_set[i]/loss[i])+"\n")