"""
DarkMatters.astro_cosmo module for calculating halo densities, magnetic fields, gas densities
"""
import numpy as np
from . import cosmology
from scipy.integrate import simpson as integrate
from scipy.optimize import bisect


def halo_density_builder(halo_dict):
    """
    Returns a lambda function for DM density rho(r)

    Arguments
    ---------------------------
    halo_dict : dictionary
        Halo properties

    Returns
    ---------------------------
    rho(r) : lambda function
        Returns DM density function, units Msun/Mpc^3
    """
    if halo_dict['profile'] == "nfw":
        return lambda x: halo_dict['rho_norm']/(x/halo_dict['scale'])/(1+x/halo_dict['scale'])**2
    elif halo_dict['profile'] == "burkert":
        return lambda x: halo_dict['rho_norm']/(1+x/halo_dict['scale'])/(1+(x/halo_dict['scale'])**2)
    elif halo_dict['profile'] == "gnfw":
        return lambda x: halo_dict['rho_norm']/(x/halo_dict['scale'])**halo_dict['index']/(1+x/halo_dict['scale'])**(3-halo_dict['index'])
    elif halo_dict['profile'] == "einasto":
        return lambda x: halo_dict['rho_norm']*np.exp(-2/halo_dict['index']*((x/halo_dict['scale'])**halo_dict['index']-1))
    elif halo_dict['profile'] == "isothermal":
        return lambda x: halo_dict['rho_norm']/(1+(x/halo_dict['scale'])**2)
    elif halo_dict['profile'] == "cgnfw":
        return lambda x: halo_dict['rho_norm']*((x+halo_dict['core_scale'])/halo_dict['scale'])**(-halo_dict['index'])*(1+x/halo_dict['scale'])**(halo_dict['index']-3)
    else:
        return None

def magnetic_field_builder(mag_dict):
    """
    Returns a lambda function for magnetic field strength B(r)

    Arguments
    ---------------------------
    mag_dict : dictionary
        Magnetic field properties

    Returns
    ---------------------------
    B(r) : lambda function
        Returns magnetic field strength function, units uG
    """
    if mag_dict['profile'] in ["pl","powerlaw"]:
        return lambda x: mag_dict['mag_norm']*(x/mag_dict['scale'])**mag_dict['index']
    elif mag_dict['profile'] == "beta":
        return lambda x: mag_dict['mag_norm']*(1 +(x/mag_dict['scale'])**2)**(3*mag_dict['index']/2)
    elif mag_dict['profile'] == "doublebeta":
        return lambda x: mag_dict['mag_norm']*(1 +(x/mag_dict['scale'])**2)**(3*mag_dict['index']/2) + mag_dict['mag_norm2']*(1 +(x/mag_dict['scale2'])**2)**(3*mag_dict['index2']/2)
    elif mag_dict['profile'] == "exp":
        return lambda x: mag_dict['mag_norm']*np.exp(-x/mag_dict['scale'])
    elif mag_dict['profile'] == "m31":
        return lambda x: (mag_dict['mag_norm']*mag_dict['scale'] + 64e-3)/(mag_dict['scale'] + x)
    elif mag_dict['profile'] == "flat":
        return lambda x: mag_dict['mag_norm']*np.ones_like(x)
    else:
        return None


def gas_density_builder(gas_dict):
    """
    Returns a lambda function for ambient gas number density n(r)

    Arguments
    ---------------------------
    gas_dict : dictionary
        Ambient gas properties

    Returns
    ---------------------------
    n(r) : lambda function
        Returns gas number density function, units 1/cm^3
    """
    if gas_dict['profile'] in ["pl","powerlaw"]:
        return lambda x: gas_dict['gas_norm']*(x/gas_dict['scale'])**gas_dict['index']
    elif gas_dict['profile'] == "beta":
        return lambda x: gas_dict['gas_norm']*(1 +(x/gas_dict['scale'])**2)**(3*gas_dict['index']/2)
    elif gas_dict['profile'] == "doublebeta":
        return lambda x: gas_dict['gas_norm']*(1 +(x/gas_dict['scale'])**2)**(3*gas_dict['index']/2) + gas_dict['gas_norm2']*(1 +(x/gas_dict['scale2'])**2)**(3*gas_dict['index2']/2)
    elif gas_dict['profile'] == "exp":
        return lambda x: gas_dict['gas_norm']*np.exp(-x/gas_dict['scale'])
    elif gas_dict['profile'] == "flat":
        return lambda x: gas_dict['gas_norm']*np.ones_like(x)
    else:
        return None

def rvir_from_rho(halo_dict,cosmo):
    """
    Returns rvir from a density profile rho(r)

    Arguments
    ---------------------------
    halo_dict : dictionary
        DM halo properties

    Returns
    ---------------------------
    rvir : float
        Virial radius [Mpc]
    """
    def average_rho(rmax,halo_dict,target=0.0):
        """
        Returns average DM density over radius rmax

        Arguments
        ---------------------------
        rmax : float
            Radius for averaging DM density [Mpc]
        halo_dict : dictionary
            DM halo properties
        target : float, optional
            Target density contrast [Msun/Mpc^3]

        Returns
        ---------------------------
        rhobar : float
            Average DM density, within rmax, - target [Msun/Mpc^3]
        """
        r_set = np.logspace(np.log10(halo_dict['scale']*1e-7),np.log10(rmax),num=100)
        rho = halo_density_builder(halo_dict)(r_set)
        return integrate(y=r_set**2*rho,x=r_set)/integrate(y=r_set**2,x=r_set)-target
    target = cosmology.delta_c(halo_dict['z'],cosmo)*cosmology.rho_crit(halo_dict['z'],cosmo) #density contast we need
    return bisect(average_rho,halo_dict['scale'],halo_dict['scale']*1e6,args=(halo_dict,target))

def rho_virial_int(halo_dict):
    """
    Returns mass within virial radius

    Arguments
    ---------------------------
    halo_dict : dictionary
        DM halo properties

    Returns
    ---------------------------
    mvir : float
        Virial mass [Msun]
    """
    r_set = np.logspace(np.log10(halo_dict['scale']*1e-7),np.log10(halo_dict['rvir']),num=100)
    if not 'rho_norm' in halo_dict.keys():
        halo_dict['rho_norm'] = 1.0
    rho = halo_density_builder(halo_dict)(r_set)
    return 4*np.pi*integrate(y=r_set**2*rho,x=r_set)