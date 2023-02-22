import numpy as np
try:
    import cosmology
except:
    import dark_matters.astro_cosmo.cosmology as cosmology
from scipy.integrate import simps as integrate
from scipy.optimize import bisect


def haloDensityBuilder(haloDict):
    """
    Returns a lambda function for DM density rho(r)

    Arguments
    ---------------------------
    haloDict : dictionary
        Halo properties

    Returns
    ---------------------------
    rho(r) : lambda function
        Returns DM density function, units Msun/Mpc^3
    """
    if haloDict['haloProfile'] == "nfw":
        return lambda x: haloDict['haloNorm']/(x/haloDict['haloScale'])/(1+x/haloDict['haloScale'])**2
    elif haloDict['haloProfile'] == "burkert":
        return lambda x: haloDict['haloNorm']/(1+x/haloDict['haloScale'])/(1+(x/haloDict['haloScale'])**2)
    elif haloDict['haloProfile'] == "gnfw":
        return lambda x: haloDict['haloNorm']/(x/haloDict['haloScale'])**haloDict['haloIndex']/(1+x/haloDict['haloScale'])**(3-haloDict['haloIndex'])
    elif haloDict['haloProfile'] == "einasto":
        return lambda x: haloDict['haloNorm']*np.exp(-2/haloDict['haloIndex']*((x/haloDict['haloScale'])**haloDict['haloIndex']-1))
    elif haloDict['haloProfile'] == "isothermal":
        return lambda x: haloDict['haloNorm']/(1+(x/haloDict['haloScale'])**2)
    elif haloDict['haloProfile'] == "cgnfw":
        return lambda x: haloDict['haloNorm']*((x+haloDict['haloCoreScale'])/haloDict['haloScale'])**(-haloDict['haloIndex'])*(1+x/haloDict['haloScale'])**(haloDict['haloIndex']-3)
    else:
        return None

def magneticFieldBuilder(magDict):
    """
    Returns a lambda function for magnetic field strength B(r)

    Arguments
    ---------------------------
    magDict : dictionary
        Magnetic field properties

    Returns
    ---------------------------
    B(r) : lambda function
        Returns magnetic field strength function, units uG
    """
    if magDict['magProfile'] in ["pl","powerlaw"]:
        return lambda x: magDict['magNorm']*(x/magDict['magScale'])**magDict['magIndex']
    elif magDict['magProfile'] == "beta":
        return lambda x: magDict['magNorm']*(1 +(x/magDict['magScale'])**2)**(3*magDict['magIndex']/2)
    elif magDict['magProfile'] == "doublebeta":
        return lambda x: magDict['magNorm']*(1 +(x/magDict['magScale'])**2)**(3*magDict['magIndex']/2) + magDict['magNorm2']*(1 +(x/magDict['magScale2'])**2)**(3*magDict['magIndex2']/2)
    elif magDict['magProfile'] == "exp":
        return lambda x: magDict['magNorm']*np.exp(1.0)**(-x/magDict['magScale'])
    elif magDict['magProfile'] == "m31":
        return lambda x: (magDict['magNorm']*magDict['magScale'] + 64e-3)/(magDict['magScale'] + x)
    elif magDict['magProfile'] == "flat":
        return lambda x: magDict['magNorm']*np.ones_like(x)
    else:
        return None


def gasDensityBuilder(gasDict):
    """
    Returns a lambda function for ambient gas number density n(r)

    Arguments
    ---------------------------
    gasDict : dictionary
        Ambient gas properties

    Returns
    ---------------------------
    n(r) : lambda function
        Returns gas number density function, units 1/cm^3
    """
    if gasDict['gasProfile'] in ["pl","powerlaw"]:
        return lambda x: gasDict['gasNorm']*(x/gasDict['gasScale'])**gasDict['gasIndex']
    elif gasDict['gasProfile'] == "beta":
        return lambda x: gasDict['gasNorm']*(1 +(x/gasDict['gasScale'])**2)**(3*gasDict['gasIndex']/2)
    elif gasDict['gasProfile'] == "doublebeta":
        return lambda x: gasDict['gasNorm']*(1 +(x/gasDict['gasScale'])**2)**(3*gasDict['gasIndex']/2) + gasDict['gasNorm2']*(1 +(x/gasDict['gasScale2'])**2)**(3*gasDict['gasIndex2']/2)
    elif gasDict['gasProfile'] == "exp":
        return lambda x: gasDict['gasNorm']*np.exp(-x/gasDict['gasScale'])
    elif gasDict['gasProfile'] == "flat":
        return lambda x: gasDict['gasNorm']*np.ones_like(x)
    else:
        return None

def rvirFromRho(haloDict,cosmo):
    """
    Returns rvir from a density profile rho(r)

    Arguments
    ---------------------------
    haloDict : dictionary
        DM halo properties

    Returns
    ---------------------------
    rvir : float
        Virial radius [Mpc]
    """
    def averageRho(rmax,haloDict,target=0.0):
        """
        Returns average DM density over radius rmax

        Arguments
        ---------------------------
        rmax : float
            Radius for averaging DM density [Mpc]
        haloDict : dictionary
            DM halo properties
        target : float, optional
            Target density contrast [Msun/Mpc^3]

        Returns
        ---------------------------
        rhobar : float
            Average DM density, within rmax, - target [Msun/Mpc^3]
        """
        r_set = np.logspace(np.log10(haloDict['haloScale']*1e-7),np.log10(rmax),num=100)
        rho = haloDensityBuilder(haloDict)(r_set)
        return integrate(r_set**2*rho,r_set)/integrate(r_set**2,r_set)-target
    target = cosmology.delta_c(haloDict['haloZ'],cosmo)*cosmology.rho_crit(haloDict['haloZ'],cosmo) #density contast we need
    #print(average_rho(rc,rhos,rc,dmmod))
    #print(average_rho(rc*30,rhos,rc,dmmod))
    return bisect(averageRho,haloDict['haloScale'],haloDict['haloScale']*1e6,args=(haloDict,target))

def rhoVirialInt(haloDict):
    """
    Returns mass within virial radius

    Arguments
    ---------------------------
    haloDict : dictionary
        DM halo properties

    Returns
    ---------------------------
    mvir : float
        Virial mass [Msun]
    """
    r_set = np.logspace(np.log10(haloDict['haloScale']*1e-7),np.log10(haloDict['haloRvir']),num=100)
    if not 'haloNorm' in haloDict.keys():
        haloDict['haloNorm'] = 1.0
    rho = haloDensityBuilder(haloDict)(r_set)
    return 4*np.pi*integrate(r_set**2*rho,r_set)