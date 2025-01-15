"""
DarkMatters.emissions module for calculating multi-frequency emissivities
"""
import numpy as np
from scipy.integrate import simpson as integrate
from scipy.special import gamma as gamma_spec
from astropy import constants,units
from .progress_bar import progress
import sys,warnings
import scipy.interpolate as sp

def radio_em_grid(electrons,f_sample,r_sample,g_sample,b_sample,ne_sample):
    """
    Radio emisivity fully vectorised

    Arguments
    ---------------------------
    electrons : float, array-like (len(r_sample),len(g_sample))
        Electron distribution 
    f_sample : float, array-like
        Frequencies for calculation [MHz]
    r_sample : float array-like
        Radial positions for electrons [Mpc]
    g_sample : float, array-like
        E/m_e for annihilation/decay yields [unit-free]
    b_sample : float, array-like (len(r_sample))
        Magnetic field [uG]
    ne_sample : float, array-like (len(r_sample))
        Gas density [cm^-3]

    Returns
    ---------------------------
    em_grid : float, array-like (len(f_sample),len(r_sample)) 
        Radio emissivity [cm^-3 s^-1]
    """
    def int_bessel(t):
        """
        Bessel integral approximation, taken from 1301.6908

        Arguments
        ---------------------------
        t : float, array-like
            Argument of Bessel function

        Returns
        ---------------------------
        bessel : float, array-like (len(t))
            Value of Bessel function at t
        """
        ak_1 = np.array([-0.97947838884478688,-0.83333239129525072,0.15541796026816246])
        ak_2 = np.array([-4.69247165562628882e-2,-0.70055018056462881,1.03876297841949544e-2])
        H1 = ak_1[0]*t + ak_1[1]*np.sqrt(t) + ak_1[2]*t**(1.0/3)
        H2 = ak_2[0]*t + ak_2[1]*np.sqrt(t) + ak_2[2]*t**(1.0/3)
        A1 = np.pi*2**(5.0/3)/np.sqrt(3.0)/gamma_spec(1.0/3)*t**(1.0/3)
        A2 = np.sqrt(0.5*np.pi*t)*np.exp(-t)
        return A1*np.exp(H1) + A2*(1-np.exp(H2))

    k = len(g_sample) #number of E bins
    num = len(f_sample)  #number of frequency sampling points
    ntheta = 61   #angular integration points
    theta_set = np.linspace(1e-2,np.pi,num=ntheta)  #choose angles 0 -> pi

    r0 = 2.82e-13  #classical electron radius (cm)
    me = (constants.m_e*constants.c**2).to("GeV").value  #electron mass (GeV)
    c = constants.c.to("cm/s").value     #speed of light (cm s^-1)
    if k < 101:
        print(g_sample[0],g_sample[-1])
        intp_elec = sp.RegularGridInterpolator((g_sample,r_sample),electrons,bounds_error=False,fill_value=0.0)
        g_sample = np.logspace(np.log10(g_sample[0]),np.log10(g_sample[-1]),num=101)
        k = len(g_sample) #number of E bins
        g_grid,r_grid = np.meshgrid(g_sample,r_sample,indexing="ij")
        print(np.min(g_grid),np.max(g_grid))
        electrons_new = intp_elec((g_grid,r_grid))
    else:
        electrons_new = electrons

    nu_grid,r_grid,e_grid,t_grid = np.meshgrid(f_sample,r_sample,g_sample,theta_set,indexing="ij")
    b_grid = np.tensordot(np.tensordot(np.ones(num),b_sample,axes=0),np.ones((k,ntheta)),axes=0)
    ne_grid = np.tensordot(np.tensordot(np.ones(num),ne_sample,axes=0),np.ones((k,ntheta)),axes=0)
    electron_grid = np.tensordot(np.tensordot(np.ones(num),electrons_new.transpose(),axes=0),np.ones(ntheta),axes=0)
    nu0 = 2.8*b_grid*1e-6      #non-relativistic gyro freq in MHz
    nup = 8980.0*np.sqrt(ne_grid)*1e-6
    with np.errstate(divide="ignore"):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'overflow')
            x = 2.0*nu_grid/(3.0*nu0*e_grid**2)*(1+(e_grid*nup/nu_grid)**2)**1.5
    a = 2.0*np.pi*np.sqrt(3.0)*r0*me/c*1e6*nu0
    with np.errstate(invalid="ignore",over="ignore"):
        p_grid_full = a*electron_grid*0.5*np.sin(t_grid)*int_bessel(x/np.sin(t_grid))
    e_grid_s = np.tensordot(np.ones((num,len(r_sample))),g_sample,axes=0) #for integration once theta is integrated out
    em_grid = integrate(y=integrate(y=p_grid_full,x=t_grid),x=e_grid_s)
    return 2*np.where(np.isnan(em_grid),0.0,em_grid) #GeV cm^-3

def primary_em_high_e(mx,rho_sample,z,g_sample,q_sample,f_sample,mode_exp):
    """
    High energy emisivity from primary gamma-rays

    Arguments
    ---------------------------
    mx : float 
        WIMP mass [GeV]
    rho_sample : float, array-like 
        Dark matter density [Msun Mpc^-3]
    z : float
        Redshift
    g_sample : float, array-like
        E/m_e for annihilation/decay yields [unit-free]
    q_sample :  float, array-like 
        Particle yields dN/dE*m_e [unit-free]
    f_sample : float, array-like
        Frequencies for calculation [MHz]
    mode_exp : float
        Falg, 1.0 for decay and 2.0 for annihilation

    Returns
    ---------------------------
    em : float, array-like (len(f_sample),len(rho_sample)) 
        Primary gamma emissivity [cm^-3 s^-1]
    """
    n = len(rho_sample)
    num = len(f_sample)
    h = constants.h.to('GeV s').value
    me = (constants.m_e*constants.c**2).to('GeV').value #electron mass (GeV)
    #msun converted to kg, convert to GeV, convert Mpc to cm 
    unit_factor = (1*units.Unit("Msun/Mpc^3")*constants.c**2).to("GeV/cm^3").value
    nwimp0 = unit_factor**mode_exp/mode_exp*(1.0/mx)**mode_exp  #non-thermal wimp density [cm^-3] (central)
    rhodm = nwimp0*rho_sample**mode_exp
    em = np.zeros((num,n),dtype=float)
    Q_func = sp.interp1d(g_sample,q_sample,fill_value=0.0,bounds_error=False)
    e_grid = np.tensordot(h*f_sample*1e6*(1+z)/me,np.ones_like(rhodm),axes=0)
    rho_grid = np.tensordot(np.ones_like(f_sample),rhodm,axes=0)
    em = Q_func(e_grid)*e_grid*rho_grid
    em = np.where(np.logical_or(e_grid<g_sample[0],e_grid>g_sample[-1]),0.0,em)
    # for i in range(0,num):
    #     E_g = h*f_sample[i]*1e6*(1+z)/me
    #     #Q_set = np.where(phys.gamma_spectrum[0] < E_g,0.0,phys.gamma_spectrum[1])
    #     #em[i,:] = integrate(Q_set,phys.gamma_spectrum[0])*rhodm[:] 
    #     if E_g < g_sample[0] or E_g > g_sample[-1]:
    #         em[i,:] = np.zeros(len(rhodm))
    #     else:
    #         em[i,:] = Q_func(E_g)*rhodm[:]*E_g #now in units of (cm^-3 s^-1) when including a factor sigmaV or Gamma
    #     progress(i+1,num)
    # sys.stdout.write("\n")
    return 2.0*em*h #2 gamma-rays per event - h converts to GeV cm^-3

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
    def g_fac(q,L):
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
    G = g_fac(q,Le)
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

def secondary_em_high_e(electrons,z,g_sample,f_sample,ne_sample,photon_temp):
    """
    High-energy emisivity from ICS and Bremstrahlung

    Arguments
    ---------------------------
    electrons : float, array-like (len(ne_sample),len(g_sample))
        Electron distribution 
    z : float
        redshift
    g_sample : float, array-like
        E/m_e for annihilation/decay yields [unit-free]
    f_sample : float, array-like
        Frequencies for calculation [MHz]
    ne_sample : float, array-like (len(r_sample))
        Gas density [cm^-3]

    Returns
    ---------------------------
    em : float, array-like (len(f_sample),len(ne_sample))
        Secondary gamma emissivity [cm^-3 s^-1]
    """
    n = len(ne_sample) #number of r shells
    k = len(g_sample) #number of E bins
    num = len(f_sample)  #number of frequency sampling points
    ntheta = 101   #angular integration points

    em = np.zeros((num,n),dtype=float)   #emisivity
    P_IC = np.zeros((num,k),dtype=float)
    P_B = np.zeros((num,k),dtype=float)
    e_int = np.zeros(ntheta,dtype=float) #angular integral sampling
    int_1 = np.zeros(k,dtype=float) #energy integral sampling

    c = constants.c.to('cm/s').value     #speed of light (cm s^-1)
    h = constants.h.to('GeV s').value
    me = (constants.m_e*constants.c**2).to('GeV').value #electron mass (GeV)
    if photon_temp == 2.7255:
        photon_temp *= (1+z)
    for i in range(0,num):  #loop over freq
        nu = f_sample[i]*(1+z) 
        E_g = nu*h*1e6 #MHz to GeV
        for l in range(0,k):   #loop over energy
            g = g_sample[l]
            if(E_g > me*g):
                P_IC[i][l] = 0.0
                P_B[i][l] = 0.0
            else:
                emax = E_g*g*me/(me*g - E_g)
                emin = emax/(4*g**2)
                e_set = np.logspace(np.log10(emin),np.log10(emax),num=ntheta)
                with np.errstate(invalid="ignore",over="ignore"):
                    e_int = black_body(e_set,photon_temp)*klein_nishina(E_g,e_set,g)
                P_IC[i][l] = c*E_g*integrate(y=e_int,x=e_set)
                P_B[i][l] = c*E_g*sigma_brem(E_g,g)
        progress(i+1,num*2)
    for i in range(0,num):
        for j in range(0,n):    
            int_1 = 2*electrons[:,j]*(P_IC[i,:] + P_B[i,:]*ne_sample[j])
            #integrate over energies to get emisivity
            em[i][j] = integrate(y=int_1,x=g_sample)
        progress(i+num+1,num*2)
    sys.stdout.write("\n")
    return em*h #h converts to GeV cm^-3
