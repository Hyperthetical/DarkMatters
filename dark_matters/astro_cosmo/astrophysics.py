import numpy as np
try:
    import cosmology
except:
    import dark_matters.astro_cosmo.cosmology as cosmology
from scipy.integrate import simps as integrate
from scipy.optimize import bisect

def haloDensityBuilder(haloDict):
    """
    Calculates the density profile over r_set
        ---------------------------
        Parameters
        ---------------------------
        r_set - Required : radial sample values [Mpc] (float)
        rc    - Required : halo radial scale length [Mpc] (float)
        dmmod - Required : halo profile code (int)
        alpha - Optional : Einasto parameter (float)
        ---------------------------
        Output
        ---------------------------
        Radial density profile values (float len(r_set))
    """
    if haloDict['haloProfile'] == "nfw":
        return lambda x: haloDict['haloNorm']/(x/haloDict['haloScale'])/(1+x/haloDict['haloScale'])**2
    elif haloDict['haloProfile'] == "burkert":
        return lambda x: haloDict['haloNorm']/(1+x/haloDict['haloScale'])/(1+(x/haloDict['haloScale'])**2)
    elif haloDict['haloProfile'] == "gnfw":
        return lambda x: haloDict['haloNorm']/(x/haloDict['haloScale'])**haloDict['haloIndex']/(1+x/haloDict['haloScale'])**(3-haloDict['haloIndex'])
    elif haloDict['haloProfile'] == "einasto":
        return lambda x: haloDict['haloNorm']*np.exp(-2/haloDict['haloIndex']*((x/haloDict['haloSCale'])**haloDict['haloIndex']-1))
    elif haloDict['haloProfile'] == "isothermal":
        return lambda x: haloDict['haloNorm']/(1+(x/haloDict['haloScale'])**2)
    elif haloDict['haloProfile'] == "cgnfw":
        return lambda x: haloDict['haloNorm']*((x+haloDict['haloCoreScale'])/haloDict['haloScale'])**(-haloDict['haloIndex'])*(1+x/haloDict['haloScale'])**(haloDict['haloIndex']-3)
    else:
        return None

def magneticFieldBuilder(magDict):
    if magDict['magProfile'] in ["pl","powerlaw"]:
        return lambda x: magDict['magNorm']*(x/magDict['magScale'])**magDict['magIndex']
    elif magDict['magProfile'] == "beta":
        return lambda x: magDict['magNorm']*(1 +(x/magDict['magScale'])**2)**(3*magDict['magIndex'])
    elif magDict['magProfile'] == "doublebeta":
        return lambda x: magDict['magNorm']*(1 +(x/magDict['magScale'])**2)**(3*magDict['magIndex']) + magDict['magNorm2']*(1 +(x/magDict['magScale2'])**2)**(3*magDict['magIndex2'])
    elif magDict['magProfile'] == "exp":
        return lambda x: magDict['magNorm']*np.exp(-x/magDict['magScale'])
    elif magDict['magProfile'] == "m31":
        return lambda x: (magDict['magNorm']*magDict['magScale'] + 64e-3)/(magDict['magScale'] + x)
    elif magDict['magProfile'] == "flat":
        return lambda x: magDict['magNorm']*np.ones_like(x)
    else:
        return None


def gasDensityBuilder(gasDict):
    if gasDict['gasProfile'] in ["pl","powerlaw"]:
        return lambda x: gasDict['gasNorm']*(x/gasDict['gasScale'])**gasDict['gasIndex']
    elif gasDict['gasProfile'] == "beta":
        return lambda x: gasDict['gasNorm']*(1 +(x/gasDict['gasScale'])**2)**(3*gasDict['gasIndex'])
    elif gasDict['gasProfile'] == "doublebeta":
        return lambda x: gasDict['gasNorm']*(1 +(x/gasDict['gasScale'])**2)**(3*gasDict['gasIndex']) + gasDict['gasNorm2']*(1 +(x/gasDict['gasScale2'])**2)**(3*gasDict['gasIndex2'])
    elif gasDict['gasProfile'] == "exp":
        return lambda x: gasDict['gasNorm']*np.exp(-x/gasDict['gasScale'])
    elif gasDict['gasProfile'] == "flat":
        return lambda x: gasDict['gasNorm']*np.ones_like(x)
    else:
        return None

def rvirFromRho(haloDict,cosmo):
    """
    Find the virial radius by locating r within which average rho = delta_c*rho_c
        ---------------------------
        Parameters
        ---------------------------
        z       - Required : redshift (float)
        rhos    - Required : characteristic density relative to critical value [] (float)
        rc      - Required : radial length scale [Mpc] (float)
        dmmod   - Required : halo profile code (int)
        cos_env - Required : cosmology environment (cosmology-env)
        alpha   - Optional : einasto parameter (float)
        ---------------------------
        Output
        ---------------------------
        rvir [Mpc] (float)
    """
    def averageRho(rmax,haloDict,target=0.0):
        """
        Find average density, relative to rho_crit, in a halo within rmax (subtract target value)
            ---------------------------
            Parameters
            ---------------------------
            rmax   - Required : max integration radius [Mpc] (float)
            rhos   - Required : characteristic density relative to critical value [] (float)
            rc     - Required : radial length scale [Mpc] (float)
            dmmod  - Required : halo profile code (int)
            alpha  - Optional : einasto parameter (float)
            target - Optional : differencing value
            ---------------------------
            Output
            ---------------------------
            average rhos - target [] (float)
        """
        r_set = np.logspace(np.log10(haloDict['haloScale']*1e-7),np.log10(rmax),num=100)
        rho = haloDensityBuilder(haloDict)(r_set)
        return integrate(r_set**2*rho,r_set)/integrate(r_set**2,r_set)-target
    target = cosmology.delta_c(haloDict['haloZ'],cosmo)*cosmology.rho_crit(haloDict['haloZ'],cosmo) #density contast we need
    #print(average_rho(rc,rhos,rc,dmmod))
    #print(average_rho(rc*30,rhos,rc,dmmod))
    return bisect(averageRho,haloDict['haloScale'],haloDict['haloScale']*1e2,args=(haloDict,target))

def rhoVirialInt(haloDict):
    r_set = np.logspace(np.log10(haloDict['haloScale']*1e-7),np.log10(haloDict['haloRvir']),num=100)
    if not 'haloNorm' in haloDict.keys():
        haloDict['haloNorm'] = 1.0
    rho = haloDensityBuilder(haloDict)(r_set)
    return 4*np.pi*integrate(r_set**2*rho,r_set)