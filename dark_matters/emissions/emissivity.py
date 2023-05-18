"""
DarkMatters.emissions module for calculating multi-frequency emissivities
"""
from matplotlib import units
import numpy as np
from scipy.integrate import simpson as integrate
from astropy import constants,units
from .progress_bar import progress
import sys,warnings
import scipy.interpolate as sp

def radioEmGrid(electrons,fSample,rSample,gSample,bSample,neSample):
    """
    Radio emmisivity fully vectorised

    Arguments
    ---------------------------
    electrons : float, array-like (len(rSample),len(gSample))
        Electron distribution 
    fSample : float, array-like
        Frequencies for calculation [MHz]
    rSample : float array-like
        Radial positions for electrons [Mpc]
    gSample : float, array-like
        E/m_e for annihilation/decay yields [unit-free]
    bSample : float, array-like (len(rSample))
        Magnetic field [uG]
    neSample : float, array-like (len(rSample))
        Gas density [cm^-3]

    Returns
    ---------------------------
    emGrid : float, array-like (len(fSample),len(rSample)) 
        Radio emissivity [cm^-3 s^-1]
    """
    def int_bessel(t):
        """
        Bessel integral approximation

        Arguments
        ---------------------------
        t : float, array-like
            Argument of Bessel function

        Returns
        ---------------------------
        bessel : float, array-like (len(t))
            Value of Bessel function at t
        """
        return 1.25*t**(1.0/3.0)*np.exp(-t)*(648.0+t**2)**(1.0/12.0)

    k = len(gSample) #number of E bins
    num = len(fSample)  #number of frequency sampling points
    ntheta = 61   #angular integration points
    theta_set = np.linspace(1e-2,np.pi,num=ntheta)  #choose angles 0 -> pi

    r0 = 2.82e-13  #classical electron radius (cm)
    me = (constants.m_e*constants.c**2).to("GeV").value  #electron mass (GeV)
    c = constants.c.to("cm/s").value     #speed of light (cm s^-1)
    if k < 200:
        intpElec = sp.interp2d(rSample,gSample,electrons)
        gSample = np.logspace(np.log10(gSample[0]),np.log10(gSample[-1]),num=201)
        k = len(gSample) #number of E bins
        electrons_new = intpElec(rSample,gSample)
    else:
        electrons_new = electrons

    nuGrid,rGrid,eGrid,tGrid = np.meshgrid(fSample,rSample,gSample,theta_set,indexing="ij")
    bGrid = np.tensordot(np.tensordot(np.ones(num),bSample,axes=0),np.ones((k,ntheta)),axes=0)
    neGrid = np.tensordot(np.tensordot(np.ones(num),neSample,axes=0),np.ones((k,ntheta)),axes=0)
    electronGrid = np.tensordot(np.tensordot(np.ones(num),electrons_new.transpose(),axes=0),np.ones(ntheta),axes=0)
    nu0 = 2.8*bGrid*1e-6      #non-relativistic gyro freq in MHz
    nup = 8980.0*np.sqrt(neGrid)*1e-6
    with np.errstate(divide="ignore"):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'overflow')
            x = 2.0*nuGrid/(3.0*nu0*eGrid**2)*(1+(eGrid*nup/nuGrid)**2)**1.5
    a = 2.0*np.pi*np.sqrt(3.0)*r0*me/c*1e6*nu0
    with np.errstate(invalid="ignore",over="ignore"):
        pGridFull = a*electronGrid*0.5*np.sin(tGrid)*int_bessel(x/np.sin(tGrid))
    eGridS = np.tensordot(np.ones((num,len(rSample))),gSample,axes=0) #for integration once theta is integrated out
    emGrid = integrate(integrate(pGridFull,tGrid),eGridS)
    return 2*np.where(np.isnan(emGrid),0.0,emGrid) #GeV cm^-3

def primaryEmHighE(mx,rhoSample,z,gSample,qSample,fSample,mode_exp):
    """
    High energy emmisivity from primary gamma-rays

    Arguments
    ---------------------------
    mx : float 
        WIMP mass [GeV]
    rhoSample : float, array-like 
        Dark matter density [Msun Mpc^-3]
    z : float
        Redshift
    gSample : float, array-like
        E/m_e for annihilation/decay yields [unit-free]
    qSample :  float, array-like 
        Particle yields dN/dE*m_e [unit-free]
    fSample : float, array-like
        Frequencies for calculation [MHz]
    mode_exp : float
        Falg, 1.0 for decay and 2.0 for annihilation

    Returns
    ---------------------------
    emm : float, array-like (len(fSample),len(rhoSample)) 
        Primary gamma emissivity [cm^-3 s^-1]
    """
    n = len(rhoSample)
    num = len(fSample)
    print(qSample)
    h = constants.h.to('GeV s').value
    me = (constants.m_e*constants.c**2).to('GeV').value #electron mass (GeV)
    #msun converted to kg, convert to GeV, convert Mpc to cm 
    unit_factor = (1*units.Unit("Msun/Mpc^3")*constants.c**2).to("GeV/cm^3").value
    nwimp0 = unit_factor**mode_exp/mode_exp*(1.0/mx)**mode_exp  #non-thermal wimp density [cm^-3] (central)
    rhodm = nwimp0*rhoSample**mode_exp
    emm = np.zeros((num,n),dtype=float)
    Q_func = sp.interp1d(gSample,qSample,fill_value=0.0,bounds_error=False)
    eGrid = np.tensordot(h*fSample*1e6*(1+z)/me,np.ones_like(rhodm),axes=0)
    rhoGrid = np.tensordot(np.ones_like(fSample),rhodm,axes=0)
    emm = Q_func(eGrid)*eGrid*rhoGrid
    print(Q_func(eGrid))
    emm = np.where(np.logical_or(eGrid<gSample[0],eGrid>gSample[-1]),0.0,emm)
    # for i in range(0,num):
    #     E_g = h*fSample[i]*1e6*(1+z)/me
    #     #Q_set = np.where(phys.gamma_spectrum[0] < E_g,0.0,phys.gamma_spectrum[1])
    #     #emm[i,:] = integrate(Q_set,phys.gamma_spectrum[0])*rhodm[:] 
    #     if E_g < gSample[0] or E_g > gSample[-1]:
    #         emm[i,:] = np.zeros(len(rhodm))
    #     else:
    #         emm[i,:] = Q_func(E_g)*rhodm[:]*E_g #now in units of (cm^-3 s^-1) when including a factor sigmaV or Gamma
    #     progress(i+1,num)
    # sys.stdout.write("\n")
    return 2.0*emm*h #2 gamma-rays per event - h converts to GeV cm^-3

def klein_nishina(E_g,E,g):
    """
    Klein-Nishina formula for ICS cross-section

    Arguments
    ---------------------------
    E_g : float
        Out-going photon energy [GeV] 
    E : float, array-like
        In-coming photon energy [GeV]
    g : float
        Electron gamma factor [unit-free] 

    Returns
    ---------------------------
    kn : float, array-like (len(E)) 
        Klein-Nishina cross-section value [GeV^-1 cm^2]
    """
    def G_fac(q,L):
        """
        G factor for ICS

        Arguments
        ---------------------------
        q : float, array-like
            Q factor
        L : float
            L factor
            
        Returns
        ---------------------------
        Gfac : float, array-like (len(q))
            ICS G factor value
        """
        return 2*q*np.log(q) + (1+2*q)*(1-q) + (L*q)**2*(1-q)/(2+2*L*q)
    re = 2.82e-13  #classical electron radius (cm)
    me = (constants.m_e*constants.c**2).to('GeV').value #electron mass (GeV)
    E_e = g*me
    sig_thom = 8*np.pi/3.0*re**2 
    Le = 4*E*g/me
    q = E_g*me/(4*E*g*(E_e - E_g))
    q[q>1] = 1.0
    G = G_fac(q,Le)
    return 3.0*sig_thom*0.25/(E*g**2)*G

def sigma_brem(E_g,g):
    """
    Bremsstrahlung cross-section

    Arguments
    ---------------------------
    E_g : float
        Out-going photon energy [GeV] 
    g : float
        Electron gamma factor [unit-free]

    Returns
    ---------------------------
    sigma_brem: float 
        Bremsstrahlung cross-section value [GeV^-1 cm^2]
    """
    re = 2.82e-13  #electron radius (cm)
    me = (constants.m_e*constants.c**2).to('GeV').value #electron mass (GeV)
    sig_thom = 8*np.pi/3.0*re**2 
    E = g*me
    a = 7.29735257e-3 
    E_d = 2*g*(E - E_g)/E_g
    if(E_d <= 0):
        phi_1 = 0.0 
        phi_2 = 0.0 
    else:
        phi_1 = 4*np.log(E_d) - 0.5
        phi_2 = 4*np.log(E_d) - 0.5
    return  3*a*sig_thom/(8*np.pi*E_g)*((1+(1-E_g/E)**2)*phi_1 - 2.0/3*(1-E_g/E)*phi_2)
    

def black_body(E,T):
    """
    Black-body energy density

    Arguments
    ---------------------------
    E : float
        Photon energy [GeV]
    T : float
        Temperature [K]

    Returns
    ---------------------------
    black_body: float 
        Black body photon density [GeV^-1 cm^-3]
    """
    #h = 6.62606957e-34 #h in J s
    h = constants.h.to('GeV s').value #h in GeV s
    c = constants.c.to('cm/s').value     #speed of light (cm s^-1)
    k = constants.k_B.to('GeV/K').value #k in GeV K^-1
    b = 1.0/(k*T)
    isnan = (1-np.exp(-E*b))
    return np.where(isnan == 0.0, 0.0, 2*4*np.pi*E**2/(h*c)**3*np.exp(-E*b)*(1-np.exp(-E*b))**(-1))

def secondaryEmHighE(electrons,z,gSample,fSample,neSample,photonTemp):
    """
    High-energy emmisivity from ICS and Bremstrahlung

    Arguments
    ---------------------------
    electrons : float, array-like (len(neSample),len(gSample))
        Electron distribution 
    z : float
        redshift
    gSample : float, array-like
        E/m_e for annihilation/decay yields [unit-free]
    fSample : float, array-like
        Frequencies for calculation [MHz]
    neSample : float, array-like (len(rSample))
        Gas density [cm^-3]

    Returns
    ---------------------------
    emm : float, array-like (len(fSample),len(neSample))
        Secondary gamma emissivity [cm^-3 s^-1]
    """
    n = len(neSample) #number of r shells
    k = len(gSample) #number of E bins
    num = len(fSample)  #number of frequency sampling points
    ntheta = 100   #angular integration points

    emm = np.zeros((num,n),dtype=float)   #emmisivity
    P_IC = np.zeros((num,k),dtype=float)
    P_B = np.zeros((num,k),dtype=float)
    e_int = np.zeros(ntheta,dtype=float) #angular integral sampling
    int_1 = np.zeros(k,dtype=float) #energy integral sampling

    c = constants.c.to('cm/s').value     #speed of light (cm s^-1)
    h = constants.h.to('GeV s').value
    me = (constants.m_e*constants.c**2).to('GeV').value #electron mass (GeV)
    if photonTemp == 2.7255:
        photonTemp *= (1+z)
    for i in range(0,num):  #loop over freq
        nu = fSample[i]*(1+z) 
        E_g = nu*h*1e6 #MHz to GeV
        for l in range(0,k):   #loop over energy
            g = gSample[l]
            if(E_g > me*g):
                P_IC[i][l] = 0.0
                P_B[i][l] = 0.0
            else:
                emax = E_g*g*me/(me*g - E_g)
                emin = emax/(4*g**2)
                e_set = np.logspace(np.log10(emin),np.log10(emax),num=ntheta)
                with np.errstate(invalid="ignore",over="ignore"):
                    e_int = black_body(e_set,photonTemp)*klein_nishina(E_g,e_set,g)
                P_IC[i][l] = c*E_g*integrate(e_int,e_set)
                P_B[i][l] = c*E_g*sigma_brem(E_g,g)
        progress(i+1,num*2)
    for i in range(0,num):
        for j in range(0,n):    
            int_1 = 2*electrons[:,j]*(P_IC[i,:] + P_B[i,:]*neSample[j])
            #integrate over energies to get emmisivity
            emm[i][j] = integrate(int_1,gSample)
        progress(i+num+1,num*2)
    sys.stdout.write("\n")
    return emm*h #h converts to GeV cm^-3