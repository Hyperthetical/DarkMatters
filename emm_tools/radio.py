#cython: language_level=3
import scipy.interpolate as sp
import sys
from emm_tools import fluxes
import numpy as np
from scipy.integrate import simps as integrate
from scipy.interpolate import interp1d,interp2d
from emm_tools.tools_emm import progress
from subprocess import call

def read_emm_c(infile,halo,phys,sim):
    """
    Retrieve output of c executable for radio emmissivity
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
        2D array of floats (sim.num x sim.n)
    """
    inf = open(infile,"r")
    line = inf.readline().strip().split()
    eArray = np.array(line,dtype=float)
    n = sim.n
    k = sim.num
    emm = np.zeros((k,n),dtype=float)
    for i in range(0,k):
        for j in range(0,n):
            emm[i][j] = eArray[i*n + j]
    inf.close()
    return emm

def write_emm_c(outfile,halo,phys,sim):
    """
    Write the input file for the c executable that finds radio emmissivity
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
    outf.write(str(len(phys.spectrum[0]))+" "+str(sim.n)+" "+str(sim.num)+"\n")
    for r in halo.r_sample[0]:
        outf.write(str(r)+" ")
    outf.write("\n")
    for x in phys.spectrum[0]:
        outf.write(str(x)+" ")
    outf.write("\n")
    for x in sim.f_sample:
        outf.write(str(x)+" ")
    outf.write("\n")
    for x in halo.b_sample:
        outf.write(str(x)+" ")
    outf.write("\n")
    for x in halo.ne_sample:
        outf.write(str(x)+" ")
    outf.write("\n")
    outf.write(str(halo.z)+"\n")
    for i in range(0,len(phys.spectrum[0])):
        for j in range(0,sim.n):
            outf.write(str(halo.electrons[i][j])+" ")
    outf.close()


def emm_from_c(outfile,infile,halo,phys,sim):
    """
    Prepare the input file, run the c executable that finds radio emmissivity and retrieve the output
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
        2D array of floats (sim.num x sim.n)
    """
    write_emm_c(outfile,halo,phys,sim)
    #sys.exit(2)
    try:
        call([sim.exec_emm_c+" "+outfile+" "+infile],shell=True)#,cwd=cdir)
    except FileNotFoundError:
        return None
    emmData = read_emm_c(infile,halo,phys,sim)
    #for i in range(0,len(emmData)):
    #    for j in range(0,len(emmData[0])):
    #        print(emmData[i][j])
    return emmData

def int_bessel(t):
    """
    Bessel integral approximation
        ---------------------------
        Parameters
        ---------------------------
        t - Required : float, array-like []
        ---------------------------
        Output
        ---------------------------
        1D float array-like []
    """
    return 1.25*t**(1.0/3.0)*np.exp(-t)*(648.0+t**2)**(1.0/12.0)

def int_bessel_intp(t):
    """
    Bessel integral approximation from interpolation function
        ---------------------------
        Parameters
        ---------------------------
        t - Required : float, array-like []
        ---------------------------
        Output
        ---------------------------
        1D float array-like []
    """
    interpx = np.array([1e-4,1e-3,1e-2,3e-2,1e-1,2e-1,2.8e-1,3e-1,5e-1,8e-1,1.0,2.0,3.0,5.0,1e1])
    interpy = np.array([0.0996,0.213,0.445,0.613,0.818,0.904,0.918,0.918,0.872,0.742,0.655,0.301,0.130,2.14e-2,1.92e-4])
    spline = interp1d(interpx,interpy)
    result = -1*np.ones(len(t))
    result = np.where(t>1e1,np.sqrt(np.pi*0.5)*np.exp(-t)*np.sqrt(t),result)
    result = np.where(t<1e-4,4*np.pi/np.sqrt(3.0)/2.67894*(t*0.5)**(1.0/3),result)
    #print(t,result)
    if t[np.where(result==-1.0)[0]] != []:
        result = np.where(result==-1.0,spline(t),result)
    #for i in range(0,len(t)):
    #    if t[i] >= 1e-4 and t[i] <= 1e1:
    #        result[i] = spline(t[i])
    return result


def radio_emm(halo,phys,sim,grid=True):
    """
    Radio emmissivity 
        ---------------------------
        Parameters
        ---------------------------
        halo       - Required : halo environment (halo_env)
        phys       - Required : physical environment (physical_env)
        sim        - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        2D float array (sim.num x sim.n) [cm^-3 s^-1]
    """
    if grid:
        return radioEmmGrid(halo.electrons,sim.f_sample*(1+halo.z),halo.r_sample[0],phys.spectrum[0],halo.b_sample,halo.ne_sample)
    else:
        return radioEmmLoop(halo.electrons,sim.f_sample,halo.r_sample[0],phys.spectrum[0],halo.b_sample,halo.ne_sample,phys.mx,halo.z)

def radioEmmLoop(electrons,fSample,rSample,gSample,bSample,neSample,mx,z):
    n = len(rSample) #number of r shells
    k = len(gSample) #number of E bins
    num = len(fSample) #number of frequency sampling points
    ntheta = 60   #angular integration points

    emm = np.zeros((num,n),dtype=float)   #emmisivity
    theta_set = np.zeros(ntheta,dtype=float) #angular sampling values
    theta_int = np.zeros(ntheta,dtype=float) #angular integral sampling
    int_1 = np.zeros(k,dtype=float) #energy integral sampling

    r0 = 2.82e-13  #electron radius (cm)
    me = 0.511e-3  #electron mass (GeV)
    c = 3.0e10     #speed of light (cm s^-1)

    theta_set = np.linspace(1e-2,np.pi,num=ntheta)  #choose angles 0 -> pi
    nu_cut = 1e12*mx/3e3 #MHz -> cut-off to stop synchrotron calculations works up 3 TeV m_x
    
    for i in range(0,num):  #loop over freq
        nu = fSample[i]*(1+z) 
        if nu > nu_cut*(1+z):
            emm[i,:] = np.zeros(n)[:]
        else:
            for j in range(0,n):  #loop over r
                bmu = bSample[j]
                ne = neSample[j]#*(1+halo.z)**3
                nu0 = 2.8*bmu*1e-6      #non-relativistic gyro freq
                nup = 8980.0*np.sqrt(ne)*1e-6    #plasma freq 
                a = 2.0*np.pi*np.sqrt(3.0)*r0*me/c*1e6*nu0  #gyro radius
                for l in range(0,k):   #loop over energy
                    g = gSample[l]
                    x = 2.0*nu/(3.0*nu0*g**2)*(1+(g*nup/nu)**2)**1.5 #dimensionless integration
                    theta_int = 0.5*np.sin(theta_set)*int_bessel(x/np.sin(theta_set))  #theta integrand vectorised
                    #print(theta_int)
                    #integrate over that and factor in electron densities
                    P_S = a*integrate(theta_int,theta_set)
                    int_1[l] = electrons[l][j]*P_S
                   
                #integrate over energies to get emmisivity
                emm[i][j] = integrate(int_1,gSample)
                #print(emm[i][j])
        progress(i+1,num)
    sys.stdout.write("\n")
    emm = np.where(np.isnan(emm),0.0,emm)
    return emm

def radioEmmGrid(electrons,fSample,rSample,gSample,bSample,neSample):
    """
    Radio emmissivity 
        ---------------------------
        Parameters
        ---------------------------
        halo       - Required : halo environment (halo_env)
        phys       - Required : physical environment (physical_env)
        sim        - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        2D float array (sim.num x sim.n) [cm^-3 s^-1]
    """
    k = len(gSample) #number of E bins
    num = len(fSample)  #number of frequency sampling points
    ntheta = 60   #angular integration points
    theta_set = np.linspace(1e-2,np.pi,num=ntheta)  #choose angles 0 -> pi

    r0 = 2.82e-13  #electron radius (cm)
    me = 0.511e-3  #electron mass (GeV)
    c = 2.998e10     #speed of light (cm s^-1)

    nuGrid,rGrid,eGrid,tGrid = np.meshgrid(fSample,rSample,gSample,theta_set,indexing="ij")
    bGrid = np.tensordot(np.tensordot(np.ones(num),bSample,axes=0),np.ones((k,ntheta)),axes=0)
    neGrid = np.tensordot(np.tensordot(np.ones(num),neSample,axes=0),np.ones((k,ntheta)),axes=0)
    electronGrid = np.tensordot(np.tensordot(np.ones(num),electrons.transpose(),axes=0),np.ones(ntheta),axes=0)
    nu0 = 2.8*bGrid*1e-6      #non-relativistic gyro freq
    nup = 8980.0*np.sqrt(neGrid)*1e-6
    x = 2.0*nuGrid/(3.0*nu0*eGrid**2)*(1+(eGrid*nup/nuGrid)**2)**1.5
    a = 2.0*np.pi*np.sqrt(3.0)*r0*me/c*1e6*nu0
    pGridFull = a*electronGrid*0.5*np.sin(tGrid)*int_bessel(x/np.sin(tGrid))
    eGridS = np.tensordot(np.ones((num,len(rSample))),gSample,axes=0) #for integration once theta is integrated out
    emmGrid = integrate(integrate(pGridFull,tGrid),eGridS)
    return np.where(np.isnan(emmGrid),0.0,emmGrid) #GeV cm^-3

def radio_flux(rf,halo,sim,grid=True):
    """
    Radio flux from emmissivity 
        ---------------------------
        Parameters
        ---------------------------
        rf         - Required : radial limit of flux integration (float) [Mpc]
        halo       - Required : halo environment (halo_env)
        sim        - Required : simulation environment (simulation_env)
        grid       - Optional : flag to use vectorised or loop-based calculation (bool)
        ---------------------------
        Output
        ---------------------------
        1D float array of fluxes (sim.num) [Jy]
    """
    if grid:
        return fluxes.fluxGrid(rf,halo.dl,sim.f_sample,halo.r_sample[0],halo.radio_emm,boostMod=halo.radio_boost/halo.boost)
    else:
        return fluxes.fluxLoop(rf,halo.dl,sim.f_sample,halo.r_sample[0],halo.radio_emm,boostMod=halo.radio_boost/halo.boost)

def radio_sb(nu_sb,halo,sim,deltaOmega=4*np.pi):
    """
    Radio surface brightness as a function of angular diameter
        ---------------------------
        Parameters
        ---------------------------
        halo       - Required : halo environment (halo_env)
        sim        - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        1D float array (sim.n) [Jy arcminute^-2]
    """
    return fluxes.surfaceBrightnessLoop(nu_sb,sim.f_sample,halo.r_sample[0],halo.radio_emm,deltaOmega)
