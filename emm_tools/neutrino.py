#cython: language_level=3
import scipy.interpolate as sp
import sys
from scipy.integrate import simps as integrate
from numpy import *
from emm_tools.tools_emm import progress

def nu_emm(halo,phys,sim):
    """
    Neutrino emmissivity from halo model
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
    h = 4.13566751086e-24 #h in GeV s
    me = 0.511e-3  #electron mass (GeV)
    #msun converted to kg, convert to GeV, convert Mpc to cm 
    nwimp0 = sqrt(1.458e-33)**halo.mode_exp/halo.mode_exp*(1.0/phys.mx)**halo.mode_exp  #non-thermal wimp density (cm^-3) (central)
    rhodm = nwimp0*halo.rho_dm_sample[0]
    emm = zeros((sim.num,sim.n),dtype=float)
    Q_func = sp.interp1d(phys.nu_spectrum[0],phys.nu_spectrum[1])
    for i in range(0,sim.num):
        E_g = h*sim.f_sample[i]*1e6*(1+halo.z)/me
        #Q_set = where(phys.nu_spectrum[0] < E_g,0.0,phys.nu_spectrum[1])
        #emm[i,:] = integrate(Q_set,phys.nu_spectrum[0])*rhodm[:] 
        if E_g < phys.nu_spectrum[0][0] or E_g > phys.nu_spectrum[0][len(phys.nu_spectrum[0])-1]:
            emm[i,:] = zeros(len(rhodm))
        else:
            emm[i,:] = Q_func(E_g)*rhodm[:]*E_g #now in units of flux
        progress(i+1,sim.num)
    sys.stdout.write("\n")
    halo.nu_emm = 2.0*emm #2 nu-rays per event 
    return halo.nu_emm

def nu_from_j(halo,phys,sim):
    """
    Neutrino emmissivity from J-factor
        ---------------------------
        Parameters
        ---------------------------
        halo       - Required : halo environment (halo_env)
        phys       - Required : physical environment (physical_env)
        sim        - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        1D float array (sim.num) [cm^-2 s^-1]
    """
    h = 4.13566751086e-24 #h in GeV s
    me = 0.511e-3  #electron mass (GeV)
    nwimp0 = 0.125/pi/phys.mx**2 #GeV^-2
    emm = zeros(sim.num,dtype=float)
    Q_func = sp.interp1d(phys.nu_spectrum[0],phys.nu_spectrum[1])
    for i in range(0,sim.num):
        E_g = h*sim.f_sample[i]*1e6*(1+halo.z)/me
        if E_g < phys.nu_spectrum[0][0] or E_g > phys.nu_spectrum[0][len(phys.nu_spectrum[0])-1]:
            emm[i] = 0.0
        else:
            emm[i] = Q_func(E_g)*halo.J*nwimp0*E_g #units of flux
            #print halo.J,Q_func(E_g)*nwimp0*E_g/(1+halo.z),1+halo.z
    return 2.0*emm #2 nu-rays per event 
    
def nu_flux(rf,halo,sim,nu_only):
    """
    Neutrino flux from emmissivity 
        ---------------------------
        Parameters
        ---------------------------
        rf         - Required : radial limit of flux integration (Mpc)
        halo       - Required : halo environment (halo_env)
        sim        - Required : simulation environment (simulation_env)
        nu_only    - Required : does nothing
        ---------------------------
        Output
        ---------------------------
        1D float array (sim.num) [Jy]
    """
    h = 4.13566751086e-24 #h in GeV s
    n = sim.n
    num = sim.num
    jj = zeros(n,dtype=float)   #temporary integrand array
    ff = zeros(num,dtype=float)    #flux density
    for i in range(0,num):
        if rf < 0.9*halo.r_sample[0][n-1]:
            halo_interp_nu = sp.interp1d(halo.r_sample[0],halo.nu_emm[i])
            rset = logspace(log10(halo.r_sample[0][0]),log10(rf),num=n)
            emm_r = halo_interp_nu(rset)
        else:
            emm_r = halo.nu_emm[i]
            rset = halo.r_sample[0]
        jj = rset**2*emm_r
        #flux density as a function of frequency, integrate over r to get there
        ff[i] = 4.0*pi*integrate(jj/(halo.dl**2+rset**2),rset)/4.0/pi
    ff = ff*3.09e24   #incident photon number density from Mpc cm^-3 s^-1 to cm^-2 s^-1
    ff = ff*h #flux from cm^-2 s^-1 to GeV cm^-2
    ff = ff*1.6e20 #flux from GeV cm^-2 to Jy
    #results must be multiplied by the chi-chi cross section
    return ff

def nu_sb(halo,sim):
    """
    Neutrino surface brightness as a function of angular diameter
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
    lum = zeros(sim.n,dtype=float)
    sb = zeros(sim.n,dtype=float) #surface brightness (nu,r)
    for j in range(0,sim.n):
        rprime = halo.r_sample[0][j]
        for k in range(0,sim.n):
            r = halo.r_sample[0][k]
            if(rprime >= r):
                lum[k] = 0.0
            else:
                lum[k] = halo.nu_emm_nu[k]*r/sqrt(r**2-rprime**2)
        sb[j] = 2.0*integrate(lum,halo.r_sample[0]) #the 2 comes from integrating over diameter not radius
    return sb*3.09e24*1.6e20/(4*pi)/1.1818e7 #unit conversions and adjustment to angles 
