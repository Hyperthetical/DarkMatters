#cython: language_level=3
import scipy.interpolate as sp
import sys
from emm_tools import fluxes
from scipy.integrate import simps as integrate
import numpy as np
from emm_tools.tools_emm import progress
from astropy import constants as const

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

def G_fac(q,L):
    """
    G factor for ICS
        ---------------------------
        Parameters
        ---------------------------
        q - Required : float, array-like []
        L - Required : float []
        ---------------------------
        Output
        ---------------------------
        1D float array-like []
    """
    return 2*q*np.log(q) + (1+2*q)*(1-q) + (L*q)**2*(1-q)/(2+2*L*q)

def klein_nishina(E_g,E,g):
    """
    Klein-Nishina formula for ICS cross-section
        ---------------------------
        Parameters
        ---------------------------
        E_g - Required : out-going photon energy [GeV] (float)
        E   - Required : in-coming photon energy [GeV] (array-like float)
        g   - Required : electron gamma factor [] (float)
        ---------------------------
        Output
        ---------------------------
        1D float array-like [GeV^-1 cm^2]
    """
    re = 2.82e-13  #electron radius (cm)
    me = (const.m_e*const.c**2).to('GeV').value #electron mass (GeV)
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
        ---------------------------
        Parameters
        ---------------------------
        E_g - Required : out-going photon energy [GeV] (float)
        g   - Required : electron gamma factor [] (float)
        ---------------------------
        Output
        ---------------------------
        float [GeV^-1 cm^2]
    """
    re = 2.82e-13  #electron radius (cm)
    me = (const.m_e*const.c**2).to('GeV').value #electron mass (GeV)
    sig_thom = 8*np.pi/3.0*re**2 
    E = g*me
    a = 7.29735257e-3 
    E_d = 2*g*(E - E_g)/E_g
    if(E_d <= 0):
        phi_1 = 0.0 #GET
        phi_2 = 0.0 #GET
    else:
        phi_1 = 4*np.log(E_d) - 0.5
        phi_2 = 4*np.log(E_d) - 0.5
    return  3*a*sig_thom/(8*np.pi*E_g)*((1+(1-E_g/E)**2)*phi_1 - 2.0/3*(1-E_g/E)*phi_2)
    

def black_body(E,T):
    """
    Black-body energy density
        ---------------------------
        Parameters
        ---------------------------
        E - Required : photon energy [GeV] (float)
        T - Required : temperature [K] (float)
        ---------------------------
        Output
        ---------------------------
        float [GeV^-1 cm^-3]
    """
    #h = 6.62606957e-34 #h in J s
    h = const.h.to('GeV s').value #h in GeV s
    c = const.c.to('cm/s').value     #speed of light (cm s^-1)
    k = const.k_B.to('GeV/K').value #k in GeV K^-1
    b = 1.0/(k*T)
    isnan = (1-np.exp(-E*b))
    return np.where(isnan == 0.0, 0.0, 2*4*np.pi*E**2/(h*c)**3*np.exp(-E*b)*(1-np.exp(-E*b))**(-1))

def high_E_emm(halo,phys,sim):
    """
    High energy emmisivity from ICS and Bremsstrahlung
        ---------------------------
        Parameters
        ---------------------------
        halo - Required : halo environment (halo_env)
        phys - Required : physical environment (physical_env)
        sim  - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        2D float array (sim.num x sim.n) [cm^-3 s^-1]
    """
    n = sim.n #number of r shells
    k = len(phys.spectrum[0]) #number of E bins
    num = sim.num  #number of frequency sampling points
    ntheta = 100   #angular integration points

    emm = np.zeros((num,n),dtype=float)   #emmisivity
    P_IC = np.zeros((num,k),dtype=float)
    P_B = np.zeros((num,k),dtype=float)
    e_int = np.zeros(ntheta,dtype=float) #angular integral sampling
    int_1 = np.zeros(k,dtype=float) #energy integral sampling

    c = const.c.to('cm/s').value     #speed of light (cm s^-1)
    h = const.h.to('GeV s').value
    me = (const.m_e*const.c**2).to('GeV').value #electron mass (GeV)
     
    for i in range(0,num):  #loop over freq
        nu = sim.f_sample[i]*(1+halo.z) 
        E_g = nu*h*1e6
        for l in range(0,k):   #loop over energy
            g = phys.spectrum[0][l]
            if(E_g > me*g):
                P_IC[i][l] = 0.0
                P_B[i][l] = 0.0
            else:
                emax = E_g*g*me/(me*g - E_g)
                emin = emax/(4*g**2)
                e_set = np.logspace(np.log10(emin),np.log10(emax),num=ntheta)
                e_int = black_body(e_set,2.73*(1+halo.z))*klein_nishina(E_g,e_set,g)
                P_IC[i][l] = c*E_g*integrate(e_int,e_set)
                P_B[i][l] = c*E_g*sigma_brem(E_g,g)
        progress(i+1,num*2)
    for i in range(0,num):
        for j in range(0,n):    
            int_1 = halo.electrons[:,j]*(P_IC[i,:] + P_B[i,:]*halo.ne_sample[j])
            #integrate over energies to get emmisivity
            emm[i][j] = integrate(int_1,phys.spectrum[0])
        progress(i+num+1,num*2)
    sys.stdout.write("\n")
    return emm*h #h converts to GeV cm^-3

def gamma_source(halo,phys,sim):
    """
    High energy emmisivity from direct gamma-rays via halo model
        ---------------------------
        Parameters
        ---------------------------
        halo - Required : halo environment (halo_env)
        phys - Required : physical environment (physical_env)
        sim  - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        2D float array (sim.num x sim.n) [cm^-3 s^-1]
    """
    h = const.h.to('GeV s').value
    me = (const.m_e*const.c**2).to('GeV').value #electron mass (GeV)
    #msun converted to kg, convert to GeV, convert Mpc to cm 
    nwimp0 = np.sqrt(1.458e-33)**halo.mode_exp/halo.mode_exp*(1.0/phys.mx)**halo.mode_exp  #non-thermal wimp density (cm^-3) (central)
    rhodm = nwimp0*halo.rho_dm_sample[0]
    emm = np.zeros((sim.num,sim.n),dtype=float)
    Q_func = sp.interp1d(phys.gamma_spectrum[0],phys.gamma_spectrum[1])
    for i in range(0,sim.num):
        E_g = h*sim.f_sample[i]*1e6*(1+halo.z)/me
        #Q_set = np.where(phys.gamma_spectrum[0] < E_g,0.0,phys.gamma_spectrum[1])
        #emm[i,:] = integrate(Q_set,phys.gamma_spectrum[0])*rhodm[:] 
        if E_g < phys.gamma_spectrum[0][0] or E_g > phys.gamma_spectrum[0][len(phys.gamma_spectrum[0])-1]:
            emm[i,:] = np.zeros(len(rhodm))
        else:
            emm[i,:] = Q_func(E_g)*rhodm[:]*E_g #now in units of (cm^-3 s^-1)
        progress(i+1,sim.num)
    sys.stdout.write("\n")
    halo.gamma_emm = 2.0*emm*h #2 gamma-rays per event - h converts to GeV cm^-3
    return halo.gamma_emm

def gamma_from_j(halo,phys,sim):
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
    h = const.h.to('GeV s').value
    me = (const.m_e*const.c**2).to('GeV').value #electron mass (GeV)
    nwimp0 = 0.125/np.pi/phys.mx**2 #GeV^-2
    emm = np.zeros(sim.num,dtype=float)
    Q_func = sp.interp1d(phys.gamma_spectrum[0],phys.gamma_spectrum[1])
    for i in range(0,sim.num):
        E_g = h*sim.f_sample[i]*1e6*(1+halo.z)/me
        if E_g < phys.gamma_spectrum[0][0] or E_g > phys.gamma_spectrum[0][len(phys.gamma_spectrum[0])-1]:
            emm[i] = 0.0
        else:
            emm[i] = Q_func(E_g)*halo.J*nwimp0*E_g #units of flux
            #print halo.J,Q_func(E_g)*nwimp0*E_g/(1+halo.z),1+halo.z
    return 2.0*emm #2 gamma-rays per event 

def gamma_from_d(halo,phys,sim):
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
    h = const.h.to('GeV s').value
    me = (const.m_e*const.c**2).to('GeV').value #electron mass (GeV)
    nwimp0 = 0.25/np.pi/phys.mx #GeV^-1
    emm = np.zeros(sim.num,dtype=float)
    Q_func = sp.interp1d(phys.gamma_spectrum[0],phys.gamma_spectrum[1])
    for i in range(0,sim.num):
        E_g = h*sim.f_sample[i]*1e6*(1+halo.z)/me
        if E_g < phys.gamma_spectrum[0][0] or E_g > phys.gamma_spectrum[0][len(phys.gamma_spectrum[0])-1]:
            emm[i] = 0.0
        else:
            emm[i] = Q_func(E_g)*halo.Dfactor*nwimp0*E_g #units of flux 
            #print halo.J,Q_func(E_g)*nwimp0*E_g/(1+halo.z),1+halo.z
    return 2.0*emm #2 gamma-rays per event  

def high_E_flux(rf,halo,sim,gamma_only=1,grid=True):
    """
    High energy flux from emmissivity 
        ---------------------------
        Parameters
        ---------------------------
        rf         - Required : radial limit of flux integration (float) [Mpc]
        halo       - Required : halo environment (halo_env)
        sim        - Required : simulation environment (simulation_env)
        gamma_only - Optional : 0 -> direct gamma only or 1 -> all mechanisms (int)
        grid       - Optional : flag to use vectorised or loop-based calculation (bool)
        ---------------------------
        Output
        ---------------------------
        1D float array of fluxes (sim.num) [Jy]
    """
    if gamma_only == 1:
        flux = fluxes.fluxGrid(rf,halo.dl,sim.f_sample,halo.r_sample[0],halo.he_emm)
        flux += fluxes.fluxGrid(rf,halo.dl,sim.f_sample,halo.r_sample[0],halo.gamma_emm)
    else:
        flux = fluxes.fluxGrid(rf,halo.dl,sim.f_sample,halo.r_sample[0],halo.gamma_emm)
    return flux #h is needed to fix units of the high-energy fluxes

def high_E_flux_old(rf,halo,sim,gamma_only=1):
    """
    High energy flux 
        ---------------------------
        Parameters
        ---------------------------
        rf         - Required : radial limit of flux integration (Mpc)
        halo       - Required : halo environment (halo_env)
        sim        - Required : simulation environment (simulation_env)
        gamma_only - Optional : 0 -> direct gamma only or 1 -> all mechanisms (int)
        ---------------------------
        Output
        ---------------------------
        1D float array (sim.num) [Jy]
    """
    #if gamma_only = 0 do only the gamma flux
    n = sim.n
    num = sim.num
    jj = np.zeros(n,dtype=float)   #temporary integrand array
    ff = np.zeros(num,dtype=float)    #flux density
    #print(rf)
    #print(gamma_only)
    for i in range(0,num):
        if rf < 0.9*halo.r_sample[0][n-1]:
            if gamma_only != 0:
                halo_interp_x = sp.interp1d(halo.r_sample[0],halo.he_emm[i])
            halo_interp_g = sp.interp1d(halo.r_sample[0],halo.gamma_emm[i])
            rset = np.logspace(np.log10(halo.r_sample[0][0]),np.log10(rf),num=n)
            if gamma_only == 0:
                emm_r = halo_interp_g(rset)
            else:
                emm_r = halo_interp_x(rset) + halo_interp_g(rset)
        else:
            if gamma_only == 0:
                emm_r = halo.gamma_emm[i]
            else:
                emm_r = halo.he_emm[i] + halo.gamma_emm[i]
            rset = halo.r_sample[0]
        jj = rset**2*emm_r
        #flux density as a function of frequency, integrate over r to get there
        if halo.J_flag == 0:
            ff[i] = 4.0*np.pi*integrate(jj/(halo.dl**2+rset**2),rset)/(4.0*np.pi)
        else:
            ff[i] = 4.0*np.pi*integrate(jj/(halo.dl**2+rset**2),rset)/(4.0*np.pi)
    ff = ff*3.09e24   #incident photon number density from GeV Mpc cm^-3 to GeV cm^-2 
    ff = ff*1.6e20 #flux from GeV cm^-2 to Jy
    #results must be multiplied by the chi-chi cross section
    return ff

def xray_sb(nu,halo,sim,deltaOmega=4*np.pi):
    """
    X-ray surface brightness for the given frequency nu as function of angular diameter
        ---------------------------
        Parameters
        ---------------------------
        halo       - Required : halo environment (halo_env)
        sim        - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        1D float array (sim.n) [Jy sr^-1]
    """
    return fluxes.surfaceBrightnessLoop(nu,sim.f_sample,halo.r_sample[0],halo.he_emm,deltaOmega)

def gamma_sb(nu,halo,sim,deltaOmega=4*np.pi):
    """
    Gamma-ray surface brightness for the given frequency nu as function of angular diameter
        ---------------------------
        Parameters
        ---------------------------
        halo       - Required : halo environment (halo_env)
        sim        - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        1D float array (sim.n) [Jy sr^-1]
    """
    return fluxes.surfaceBrightnessLoop(nu,sim.f_sample,halo.r_sample[0],halo.gamma_emm,deltaOmega)
