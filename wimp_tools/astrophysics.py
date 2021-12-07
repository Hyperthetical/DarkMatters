#cython: language_level=3
from scipy.integrate import quad
from scipy.optimize import newton,bisect
import numpy as np
try:
    from wimp_tools import tools,cosmology
except:
    import wimp_tools.cosmology as cosmology
    import wimp_tools.tools as tools

b_model_set = ["flat","powerlaw","pl","exp","m31","m31exp","follow_ne","equipartition","sc2006","king"]
ne_model_set = ["flat","exp","pl","powerlaw","king"]

#==========================================================================================
# Tools for getting extra halo information
#==========================================================================================

def einInt(xmin,xmax,alpha):
    """
    Integrate an einasto profile 
        ---------------------------
        Parameters
        ---------------------------
        xmin  - Required : minimum value of r/rs (float)
        xmax  - Required : maximum value of r/rs (float)
        alpha - Required : Einasto alpha parameter (float)
        ---------------------------
        Output
        ---------------------------
        Integral value from xmin to xmax [] (float) 
    """
    #note the 4pi is left out
    xset = np.logspace(np.log10(xmin),np.log10(xmax),num=100)
    yset = xset**2*np.exp(-2/alpha*(xset**alpha-1))
    return tools.Integrate(yset,xset)

def get_rcore_ein(mvir,rvir,rho0,alpha):
    """
    Find rs for an einasto profile via optimisation
        ---------------------------
        Parameters
        ---------------------------
        mvir  - Required : virial mass [Msol] (float)
        rvir  - Required : virial radius [Mpc] (float)
        rho0  - Required : characteristic density [Msol Mpc^-3] (float)
        alpha - Required : Einasto alpha parameter (float)
        ---------------------------
        Output
        ---------------------------
        rs [Mpc] (float) 
    """
    def root_rcore_ein(r,rvir,rho0,mvir,alpha):
        """
        Zeros of this function locate r=rs for einasto halos
            ---------------------------
            Parameters
            ---------------------------
            r     - Required : radial position [Mpc] (float)
            rvir  - Required : virial radius [Mpc] (float)
            rho0   - Required : characteristic density [Msol Mpc^-3] (float)
            mvir  - Required : virial mass [Msol] (float)
            alpha - Required : einasto parameter (float)
            ---------------------------
            Output
            ---------------------------
            Normalisation variance from mvir [Msol] (float)
        """
        x = rvir/r
        return 4*np.pi*r**3*rho0*einInt(1e-7,x,alpha) - mvir
    #this optimises an r_s for the einasto profile with everything else specified
    return newton(root_rcore_ein,0.028*mvir/1.2e12,args=(rvir,rho0,mvir,alpha),maxiter=200) 

def getProfile_einasto(z,mvir,alpha,cos_env,cvir_match=False):
    """
    Find rho0,rs,cvir for an einasto profile 
        ---------------------------
        Parameters
        ---------------------------
        z          - Required : redshift (float)
        mvir       - Required : virial mass [Msol] (float)
        alpha      - Required : Einasto alpha parameter (float)
        cos_env    - Required : cosmology environment (cosmology_env)
        cvir_match - Optional : True use einasto cvir, False use default N-body formula (boolean)
        ---------------------------
        Output
        ---------------------------
        rho0,rs,cvir [Msol Mpc^-3,Mpc,-] (float,float,float) 
    """
    #returns a profile, the r_s value, and cvir
    r2 = r2_ein(mvir,z,alpha,cos_env,cvir_match) #find r where the profile slope is r**(-2)
    return rho_ein(mvir,z,alpha,cos_env,cvir_match),r2,cvir_ein(z,mvir,cos_env,cvir_match)

def rho_ein(mvir,z,alpha,cos_env,cvir_match=False):
    """
    Find rho0 for an einasto profile 
        ---------------------------
        Parameters
        ---------------------------
        mvir       - Required : virial mass [Msol] (float)
        rvir       - Required : virial radius [Mpc] (float)
        rho0       - Required : characteristic density [Msol Mpc^-3] (float)
        alpha      - Required : Einasto alpha parameter (float)
        cvir_match - Optional : True use einasto cvir, False use default N-body formula (boolean)
        ---------------------------
        Output
        ---------------------------
        rho0 [Msol Mpc^-3] (float) 
    """
    #this solves for the characteristic density to match mvir
    cv = cvir_ein(z,mvir,cos_env,cvir_match)
    dc = cosmology.delta_c(z,cos_env)
    rhoc = cosmology.rho_crit(z,cos_env)
    return dc*rhoc/np.exp(-2/alpha*(cv**alpha-1))

def r2_ein(mvir,z,alpha,cos_env,cvir_match=False):
    """
    Find rs for an einasto profile via fitting formulae
        ---------------------------
        Parameters
        ---------------------------
        mvir       - Required : virial mass [Msol] (float)
        z          - Required : redshift (float)
        rho0       - Required : characteristic density [Msol Mpc^-3] (float)
        alpha      - Required : Einasto alpha parameter (float)
        cos_env    - Required : cosmology environment (cosmology_env)
        cvir_match - Optional : True use einasto cvir, False use default N-body formula (boolean)
        ---------------------------
        Output
        ---------------------------
        rs [Mpc] (float) 
    """
    #this solves for r_s to match mvir
    cv = cvir_ein(z,mvir,cos_env,cvir_match)
    rho2 = rho_ein(mvir,z,alpha,cos_env,cvir_match)
    return (mvir/(4*np.pi*rho2)/einInt(1e-10,cv,alpha))**(1.0/3)

def mstar(z,cos_env):
    """
    Fitting value from 1804.10199
        ---------------------------
        Parameters
        ---------------------------
        z          - Required : redshift (float)
        cos_env    - Required : cosmology environment (cosmology_env)
        ---------------------------
        Output
        ---------------------------
        Mstar [Msol] (float)
    """
    #returns mstar in solar masses
    return 268337.28/cos_env.h*10**(-0.651442*z) #fitted from 1804.10199

def cvir_ein(z,mvir,cos_env,cvir_match=False):
    """
    Find cvir for einasto halos from stacked analysis fitting
        ---------------------------
        Parameters
        ---------------------------
        z          - Required : redshift (float)
        mvir       - Required : virial mass [Msol] (float)
        cos_env    - Required : cosmology environment (cosmology_env)
        cvir_match - Optional : True use einasto cvir, False use default N-body formula (boolean)
        ---------------------------
        Output
        ---------------------------
        cvir [] (float) 
    """
    #1804.10199 stacked analysis
    #this is a fitting function for einasto halos
    #returns virial concentration
    A = 63.2
    c0 = 3.36
    m = -0.01
    mt = mstar(z,cos_env)*431.48
    if not cvir_match:
        return A*((mvir/mt)**m*(1+mvir/mt)**(-m) -1) + c0
    else:
        return cosmology.cvir_p12(mvir,z,cos_env)

def get_rvir_ein(rs,mvir,rho0,alpha):
    """
    Find rvir for an einasto halo by optimisation
        ---------------------------
        Parameters
        ---------------------------
        r     - Required : radial position [Mpc] (float)
        mvir  - Required : virial mass [Msol] (float)
        rho0   - Required : characteristic density [Msol Mpc^-3] (float)
        alpha - Required : einasto parameter (float)
        ---------------------------
        Output
        ---------------------------
        rvir [Mpc] (float)
    """
    def root_rvir_ein(r,rs,rho0,mvir,alpha):
        """
        Zeros of this function locate r=rvir for einasto halos
            ---------------------------
            Parameters
            ---------------------------
            r     - Required : radial position [Mpc] (float)
            rs    - Required : radial scale length [Mpc] (float)
            rho0   - Required : characteristic density [Msol Mpc^-3] (float)
            mvir  - Required : virial mass [Msol] (float)
            alpha - Required : einasto parameter (float)
            ---------------------------
            Output
            ---------------------------
            Normalisation variance from mvir [Msol] (float)
        """
        x = r/rs
        return 4*np.pi*rs**3*rho0*einInt(1e-7,x,alpha) - mvir
    return newton(root_rvir_ein,0.35*mvir/1.2e12,args=(rs,rho0,mvir,alpha))


def get_rcore_burkert(rvir,mvir):
    """
    Find rs for a burkert halo by optimisation
        ---------------------------
        Parameters
        ---------------------------
        rvir  - Required : virial radius [Mpc] (float)
        mvir  - Required : virial mass [Msol] (float)
        ---------------------------
        Output
        ---------------------------
        rs [Mpc] (float)
    """
    def root_rcore_burkert(r,rvir,mvir):
        """
        Zeros of this function locate r=rs for burkert halos
            ---------------------------
            Parameters
            ---------------------------
            r     - Required : radial position [Mpc] (float)
            rvir  - Required : virial radius [Mpc] (float)
            mvir  - Required : virial mass [Msol] (float)
            ---------------------------
            Output
            ---------------------------
            Normalisation variance from mvir [Msol] (float)
        """
        x = rvir/r
        return 0.25*(np.log(x**2+1)+2*np.log(1+x)-2*np.arctan(x))*4*np.pi*r**3*rhos_burkert(r) - mvir
    return newton(root_rcore_burkert,0.028*mvir/1.5e12,args=(rvir,mvir),maxiter=200)

def get_rvir_burkert(rb,mvir):
    """
    Find rvir for a burkert halo by optimisation
        ---------------------------
        Parameters
        ---------------------------
        rb    - Required : radial scale length [Mpc] (float)
        mvir  - Required : virial mass [Msol] (float)
        ---------------------------
        Output
        ---------------------------
        rvir [Mpc] (float)
    """
    def root_rvir_burkert(r,rb,rho0,mvir):
        """
        Zeros of this function locate r=rvir for burkert halos
            ---------------------------
            Parameters
            ---------------------------
            r     - Required : radial position [Mpc] (float)
            rb    - Required : radial scale [Mpc] (float)
            rho0   - Required : characteristic density [Msol Mpc^-3] (float)
            mvir  - Required : virial mass [Msol] (float)
            ---------------------------
            Output
            ---------------------------
            Normalisation variance from mvir [Msol] (float)
        """
        x = r/rb
        return 0.25*(np.log(x**2+1)+2*np.log(1+x)-2*np.arctan(x))*4*np.pi*rb**3*rho0 - mvir
    rho0 = rhos_burkert(rb)
    #print(rho0)
    return newton(root_rvir_burkert,0.15*mvir/1.5e12,args=(rb,rho0,mvir),maxiter=200)

def rhos_burkert(rb):
    """
    Find rho0 relative to critical density for a burkert halo by fitting function
        ---------------------------
        Parameters
        ---------------------------
        rb  - Required : radial scale length [Mpc] (float)
        ---------------------------
        Output
        ---------------------------
        rhos [] (float)
    """
    return 4.3e16/1.6/(rb*1e3)**(2.0/3) #empirical relation from Burkert 1995

def jfactor(halo,cos_env):
    """
    Find jfactor for given halo
        ---------------------------
        Parameters
        ---------------------------
        halo    - Required : halo environment (halo_env)
        cos_env - Required : cosmology environment (cosmology_env)
        ---------------------------
        Output
        ---------------------------
        jfactor [GeV^2 cm^-5] (float)
    """
    junits = 4.428e-9 #from M_sun^2 Mpc^-5 to Gev^2 cm^-5
    if halo.mode_exp == 1.0:
        jexp = 2.0
    else:
        jexp = 1.0
    jfac = tools.Integrate((halo.rho_dm_sample[0])**jexp*halo.r_sample[0]**2/(halo.r_sample[0]**2 + halo.dl**2),halo.r_sample[0])*junits
    return jfac

def dfactor(halo,cos_env):
    """
    Find dfactor for given halo
        ---------------------------
        Parameters
        ---------------------------
        halo    - Required : halo environment (halo_env)
        cos_env - Required : cosmology environment (cosmology_env)
        ---------------------------
        Output
        ---------------------------
        dfactor [GeV cm^-2] (float)
    """
    mSolToGeV = 2e30*2.998e8**2*6.242e9 #GeV Msol^-1
    mpcToCm = 3.086e24 #cm Mpc^-1
    dunits = mSolToGeV/mpcToCm**2 #from M_sun Mpc^-2 to Gev cm^-2
    if halo.mode_exp == 1.0:
        dexp = 1.0
    else:
        dexp = 0.5
    dfac = tools.Integrate((halo.rho_dm_sample[0])**dexp*halo.r_sample[0]**2/(halo.r_sample[0]**2 + halo.dl**2),halo.r_sample[0])*dunits
    return dfac

def average_rho(rmax,rhos,rc,dmmod,alpha=0.17,target=0.0):
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
    n = 100
    r_set = np.logspace(np.log10(rc*1e-7),np.log10(rmax),num=n)
    rho = rho_dm(r_set,rc,dmmod,alpha)*rhos
    return tools.Integrate(r_set**2*rho,r_set)/tools.Integrate(r_set**2,r_set)-target

def rvir_from_rho(z,rhos,rc,dmmod,cos_env,alpha=0.17):
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
    target = cosmology.delta_c(z,cos_env) #density contast we need
    #print(average_rho(rc,rhos,rc,dmmod))
    #print(average_rho(rc*30,rhos,rc,dmmod))
    return bisect(average_rho,rc,rc*1e2,args=(rhos,rc,dmmod,alpha,target))

#==========================================================================================
# Density profile computation
#==========================================================================================

def rho_nfw_core(halo,cos_env):
    """
    Experimental method for cored nfw halo, this is formed via a star-formation-related baryonic feedback mechanism
        ---------------------------
        Parameters
        ---------------------------
        halo    - Required : halo environment (halo_env)
        cos_env - Required : cosmology environment (cosmology-env)
        ---------------------------
        Output
        ---------------------------
        Radial density profile list [r_sample[0],r_sample[1]] (float sim.n,sim.ngr)
    """
    if halo.r_stellar_half_light is None or halo.t_sf is None:
        tools.fatal_error("Please specify stellar half-light radius and star-formation time to use the \'nfwcore\' profile")
    eta = 1.75 #1508.04143
    kappa = 0.04 #1508.04143
    rs = halo.rcore
    rmin = halo.r_sample[0][0]
    rset = np.logspace(np.log10(rmin),np.log10(rs),num=50)
    rho_nfw = halo.rhos*rho_dm(rset,rs,1)
    M_rs = tools.Integrate(rho_nfw*rset**2,rset)*4*np.pi
    rc = eta*halo.r_stellar_half_light
    tdyn = 2*np.pi*np.sqrt(rs**3/cos_env.G_newton/M_rs)
    q = halo.t_sf/tdyn*kappa
    n = np.tanh(q)
    f = [np.tanh(halo.r_sample[0]/rc),np.tanh(halo.r_sample[1]/rc)]
    rho_nfw = [rho_dm(halo.r_sample[0],rs,1),rho_dm(halo.r_sample[1],rs,1)]
    M_nfw = tools.Integrate(rho_nfw[0]*halo.r_sample[0]**2,halo.r_sample[0])*4*np.pi
    return [f[0]**n*rho_nfw[0] + n*f[0]**(n-1)*(1-f[0]**2)/4/np.pi/halo.r_sample[0]**2/rc*M_nfw,f[1]**n*rho_nfw[1] + n*f[1]**(n-1)*(1-f[1]**2)/4/np.pi/halo.r_sample[1]**2/rc*M_nfw]

def rho_nfw(halo,cos_env):
    """
    Wrapper method for NFW halo profile
        ---------------------------
        Parameters
        ---------------------------
        halo    - Required : halo environment (halo_env)
        cos_env - Required : cosmology environment (cosmology-env)
        ---------------------------
        Output
        ---------------------------
        Radial density profile list [r_sample[0],r_sample[1]] (float sim.n,sim.ngr)
    """
    rs = halo.rcore
    return [rho_dm(halo.r_sample[0],rs,halo.dm),rho_dm(halo.r_sample[1],rs,1)]

def rho_gnfw(halo,cos_env):
    """
    Wrapper method for generalised NFW halo profile
        ---------------------------
        Parameters
        ---------------------------
        halo    - Required : halo environment (halo_env)
        cos_env - Required : cosmology environment (cosmology-env)
        ---------------------------
        Output
        ---------------------------
        Radial density profile list [r_sample[0],r_sample[1]] (float sim.n,sim.ngr)
    """
    rs = halo.rcore
    return [rho_dm(halo.r_sample[0],rs,halo.gnfw_gamma),rho_dm(halo.r_sample[1],rs,halo.gnfw_gamma)]

def rho_burkert(halo,cos_env):
    """
    Wrapper method for burkert halo profile
        ---------------------------
        Parameters
        ---------------------------
        halo    - Required : halo environment (halo_env)
        cos_env - Required : cosmology environment (cosmology-env)
        ---------------------------
        Output
        ---------------------------
        Radial density profile list [r_sample[0],r_sample[1]] (float sim.n,sim.ngr)
    """
    rs = halo.rcore
    return [rho_dm(halo.r_sample[0],rs,2),rho_dm(halo.r_sample[1],rs,2)]

def rho_isothermal(halo,cos_env):
    """
    Wrapper method for isothermal halo profile
        ---------------------------
        Parameters
        ---------------------------
        halo    - Required : halo environment (halo_env)
        cos_env - Required : cosmology environment (cosmology-env)
        ---------------------------
        Output
        ---------------------------
        Radial density profile list [r_sample[0],r_sample[1]] (float sim.n,sim.ngr)
    """
    rs = halo.rcore
    return [rho_dm(halo.r_sample[0],rs,3),rho_dm(halo.r_sample[1],rs,3)]

def rho_einasto(halo,cos_env):
    """
    Wrapper method for Einasto halo profile
        ---------------------------
        Parameters
        ---------------------------
        halo    - Required : halo environment (halo_env)
        cos_env - Required : cosmology environment (cosmology-env)
        ---------------------------
        Output
        ---------------------------
        Radial density profile list [r_sample[0],r_sample[1]] (float sim.n,sim.ngr)
    """
    rs = halo.rcore
    return [rho_dm(halo.r_sample[0],rs,-1,alpha=halo.alpha),rho_dm(halo.r_sample[1],rs,-1,alpha=halo.alpha)]

def rho_dm_halo(halo,cos_env):
    """
    Calls correct density profile wrapper method for halo.profile value
        ---------------------------
        Parameters
        ---------------------------
        halo    - Required : halo environment (halo_env)
        cos_env - Required : cosmology environment (cosmology-env)
        ---------------------------
        Output
        ---------------------------
        Radial density profile list [r_sample[0],r_sample[1]] (float sim.n,sim.ngr)
    """
    if halo.profile == "nfw":
        return rho_nfw(halo,cos_env)
    elif halo.profile == "gnfw":
        return rho_gnfw(halo,cos_env)
    elif halo.profile == "burkert":
        return rho_burkert(halo,cos_env)
    elif halo.profile == "nfwcore":
        return rho_nfw_core(halo,cos_env)
    elif halo.profile == "isothermal":
        return rho_isothermal(halo,cos_env)
    elif halo.profile == "einasto":    
        return rho_einasto(halo,cos_env)
    else:
        tools.fatal_error("Halo profile "+halo.profile+" is not valid")


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
        return lambda x: haloDict['rhoNorm']/(x/haloDict['haloScale'])/(1+x/haloDict['haloScale'])**2
    elif haloDict['haloProfile'] == "burkert":
        return lambda x: haloDict['rhoNorm']/(1+x/haloDict['haloScale'])/(1+(x/haloDict['haloScale'])**2)
    elif haloDict['haloProfile'] == "gnfw":
        return lambda x: haloDict['rhoNorm']/(x/haloDict['haloScale'])**haloDict['haloIndex']/(1+x/haloDict['haloScale'])**(3-haloDict['haloIndex'])
    elif haloDict['haloProfile'] == "einasto":
        return lambda x: haloDict['rhoNorm']*np.exp(-2/haloDict['HaloAlpha']*((x/haloDict['haloSCale'])**haloDict['haloAlpha']-1))
    elif haloDict['haloProfile'] == "isothermal":
        return lambda x: haloDict['rhoNorm']/(1+(x/haloDict['haloScale'])**2)
    else:
        tools.fatal_error("haloProfile {} not recognised".format(haloDict['haloProfile']))


def rho_dm(r_set,rc,dmmod,alpha=0.17):
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
    #dmmod 1 is NFW, 2 is Burkert, 3 is Isothermal, -1 gives Einasto, other is generalised NFW
    n = len(r_set)
    rho = np.zeros(n,dtype=float)
    if(dmmod != -1 and dmmod != 4):
        x = r_set/rc
        if(dmmod == 2):
            #burkert
            rho = ((1+x)*(1+x**2))**(-1)
        elif(dmmod == 3):
            #isothermal
            rho = (1+x**2)**(-1)
        else:
            rho = 1.0/(x**(dmmod)*(1.0+x)**(3-dmmod))
    else:
        if dmmod != 4:
            x = r_set/rc
        else:
            x = abs(r_set-1e-4)/rc
        rho = np.exp(-2.0/alpha*(x**alpha -1.0))
    return rho

def rho_boost(rho,rho_sub,rhos,rhobar,rhobs,rhoc,bf,fsub,mode_exp):
    """
    Calculates the density profile modification gen a substructure boosting factor for sc2006 model
        ---------------------------
        Parameters
        ---------------------------
        rho      - Required : radial density profile unboosted (float array-like)
        rho_sub  - Required : substructure radial distribution (float array-like)
        rhos     - Required : characteristic density relative to critical value (float)
        rhobar   - Required : average density [Msol Mpc^-3] (float)
        rhobs    - Required : substructure normalisation density [Msol Mpc^-3] (float)
        rhoc     - Required : critical density [Msol Mpc^-3] (float)
        bf       - Required : boosting factor (float)
        fsub     - Required : halo mass fraction in sub-halos (float)
        mode_exp - Required : 2 for annihilation, 1 for decay (float)
        ---------------------------
        Output
        ---------------------------
        Radial density profile values ^ mode_exp (float len(rho))
    """
    n = len(rho)
    rhodm = np.zeros(n,dtype=float)
    rhodm = ((rhoc*rhos*rho - fsub*rhobs*rho_sub)**2/rhobar**2 + fsub*rhobs*rho_sub*bf/rhobar)*rhobar**2
    #print "boost:"+str((rhodm/rho**2/rhos**2/rhoc**2).sum()/n)
    return rhodm**(mode_exp*0.5) #rhodm is rho**2, which explains the mode_exp*0.5 power

def rho_volume_int(rv,rc,q,dmmod,alpha=0.18):
    """
    Calculates the volume integral over unit-free radial density profile
        ---------------------------
        Parameters
        ---------------------------
        rv    - Required : maximum radius (float)
        rc    - Required : radial length scale (float)
        q     - Required : mode_exp exponent, 2 annihilation, 1 decay (float)
        dmmod - Required : density profile code (float)
        alpha - Optional : Einasto parameter (float)
        ---------------------------
        Output
        ---------------------------
        Integral value (float)
    """
    n = 100
    r_set = np.zeros(n,dtype=float)
    int_set = np.zeros(n,dtype=float)

    r_set = np.logspace(np.log10(rc*1e-7),np.log10(rv),num=n)
    rho = rho_dm(r_set,rc,dmmod,alpha)
    int_set = r_set**2*rho**q
    I = tools.Integrate(int_set,r_set)*4.0*np.pi
    return I


#==========================================================================================
# Gas density profiles
#==========================================================================================

def king_radial_scale(r,rc,ex):
    """
    King-like radial profile (unit-free)
        ---------------------------
        Parameters
        ---------------------------
        r  - Required : radial position (array-like float)
        rc - Required : scaling radius (float)
        ex - Required : scaling exponent (String)
        ---------------------------
        Output
        ---------------------------
        King-type radial profle [] (array-like float, matching shape(r))
    """
    return (1.0+(r/rc)**2)**ex


def ne_distribution(phys,halo,ne_model):
    """
    Select and calculate gas density profile
        ---------------------------
        Parameters
        ---------------------------
        phys     - Required : physical environment (physical_env)
        halo     - Required : halo environment (halo_env)
        ne_model - Required : gas profile label (String)
        ---------------------------
        Output
        ---------------------------
        Gas density profile array-like (float len(halo.r_sample[0]))
    """
    if ne_model in ["powerlaw","pl"]:
        if phys.lb == None:
            lb = halo.rcore
        else:
            lb = phys.lb
        ne_set = phys.ne0*(halo.r_sample[0]/lb)**(-phys.qe)
    elif ne_model in ["king"]:
        if phys.lb == None:
            lb = halo.rcore
        else:
            lb = phys.lb
        ne_set = phys.ne0*king_radial_scale(halo.r_sample[0],lb,-phys.qe)
    elif ne_model == "exp":
        if phys.lb == None:
            lb = halo.r_stellar_half_light
        else:
            lb = phys.lb
        ne_set = phys.ne0*np.exp(-halo.r_sample[0]/lb)
    else:
        ne_set = phys.ne0*np.ones(len(halo.r_sample[0]))
    return np.array(ne_set)

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

#==========================================================================================
# Magnetic field profiles
#==========================================================================================

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

def bfield(phys,halo,cos_env,b_flag):
    """
    Select and calculate magnetic field strength profile
        ---------------------------
        Parameters
        ---------------------------
        phys     - Required : physical environment (physical_env)
        halo     - Required : halo environment (halo_env)
        cos_env  - Required : cosmology environment (cosmology_env)
        b_flag   - Required : magnetic field profile label (String)
        ---------------------------
        Output
        ---------------------------
        Magnetic field strength profile array-like (float len(halo.r_sample[0]))
    """
    n = len(halo.r_sample[0])
    if(b_flag in ["powerlaw","pl"]):
        #power-law scaling with radius
        if phys.lb == None:
            lb = halo.rcore
        else:
            lb = phys.lb
        b_set = phys.b0*(halo.r_sample[0]/lb)**(-phys.qb)
    elif b_flag == "king":
        #more king-like than strict power-law
        if phys.lb == None:
            lb = halo.rcore
        else:
            lb = phys.lb
        b_set = phys.b0*king_radial_scale(halo.r_sample[0],lb,-phys.qb*phys.qe)
        phys.btag = "b"+str(phys.b0)+"q"+str(phys.qb*phys.qe)+"_powerlaw"
    elif(b_flag == "follow_ne"):
        #this follows the gas density with a power-law index
        b_set = phys.b0*(halo.ne_sample/halo.ne_sample[0])**phys.qb
        phys.btag = "b"+str(phys.b0)+"q"+str(phys.qb)+"_follow_ne"
    elif(b_flag == "equipartition"):
        #this model nomalises a power-law B to the energy in the gas
        rcore_z0 = cosmology.rcore(halo.mvir,0.0,cos_env)
        rvir_z0 = cosmology.rvir(halo.mvir,0.0,cos_env)
        r_set_z0 = np.logspace(np.log10(rcore_z0*1e-7),np.log10(rvir_z0),num=n) #spherical shells within radio halo
        neav_z0 =  phys.ne0*tools.weightedVolAvg(king_radial_scale(r_set_z0,rcore_z0,-phys.qe),r_set_z0)
        b0 = normalise_b(halo.mvir,1e-3*phys.lc,rvir_z0,neav_z0,1.0/3)
        b_set = b0*equipartition_bfield_profile(halo.r_sample[0],1e-3*phys.lc,1.0/3)       #magnetic field within the halo
        phys.btag = "b"+str(b0)+"_equipartition"
    elif(b_flag == "sc2006"):
        #a model that peaks away from r=0
        b_set = phys.b0*(1+halo.r_sample[0]/halo.rb1Dist)**2*(1+halo.r_sample[0]**2/halo.rb2Dist**2)**(-phys.qb)
        phys.btag = "b"+str(phys.b0)+"q"+str(phys.qb)+"_sc2006"
    elif(b_flag == "exp"):
        #an exponentially decaying radial profile
        if phys.qb == 0.0:
            rd = halo.r_stellar_half_light
        else:
            rd = phys.qb
        b_set = phys.b0*np.ones(n)*np.exp(-halo.r_sample[0]/rd)
        phys.btag = "b"+str(phys.b0)+"_exp"
    elif(b_flag == "m31"):
        #based on Ruiz-Granados 2010 model for M31 (valid out to 40 kpc)
        if phys.qb == 0.0:
            r1 = 2e2 #kpc
        else:
            r1 = phys.qb*1e3 #kpc
        b_set = (phys.b0*r1 + 64.0)/(r1+halo.r_sample[0]*1e3) #m31, distances in kpc
        phys.btag = "b"+str(phys.b0)+"_m31"
    elif(b_flag == "m31exp"):
        #exponential extrapolation from M31 profile (r =< 40 kpc is the same as M31 above)
        if phys.qb == 0.0:
            r1 = 2e2*1e-3 #Mpc
        else:
            r1 = phys.qb #Mpc
        r_exp = 40.0e-3 #Mpc
        b_set = (phys.b0*r1 + 64.0e-3)/(r1+halo.r_sample[0]) #m31
        phys.btag = "b"+str(phys.b0)+"_m31exp"
        b_set = np.where(halo.r_sample[0]>r_exp,(phys.b0*r1 + 64.0e-3)/(r1+r_exp)*np.exp(-(halo.r_sample[0]-r_exp)/halo.r_stellar_half_light),b_set)
    else:
        #just a boring flat profile
        b_set = phys.b0*np.ones(n)
        phys.btag = "b"+str(phys.b0)+"_flat"
    return b_set

def equipartition_bfield_profile(r,rmin,w):
    """
    Radial profile for equipartition magnetic field model
        ---------------------------
        Parameters
        ---------------------------
        r    - Required : radial position [Mpc] (float array-like)
        rmin - Required : scaling radius [Mpc] (float)
        w    - Required : scaling exponent [] (float)
        ---------------------------
        Output
        ---------------------------
        g (float len(r))
    """
    g_set = (r/rmin)**(w-1.0)
    g_set[r < rmin] = 1.0
    return g_set

def normalise_b(Mvir,rmin,rvir,ne,w):
    """
    Normalisation for equipartition magnetic field model
        ---------------------------
        Parameters
        ---------------------------
        Mvir - Required : virial mass [Msol] (float)
        rmin - Required : scaling radius [Mpc] (float)
        rvir - Required : virial radius [Mpc] (float)
        ne   - Required : gas density normalisation [cm^-3] (float)
        w    - Required : scaling exponent [] (float)
        ---------------------------
        Output
        ---------------------------
        B0 [uG] (float)
    """
    mu_mol = 1.6733e-24
    scaling = 1e-1 #ratio of Ub (b field energy density) to Uth (gas enegy density)
    KbT = 0.5*mu_mol*Mvir*1.9891e30*6.674e-8/rvir*3.24077929e-25*1e3
    #KbT = (Mvir/1e-15)**(2.0/3)*delta**(1.0/3)*(1+z)*1.6021773e-9/beta
    #KbT = 2*5.8e-13*(Mvir/1e-15)**(2.0/3)*delta**(1.0/3)
    Uth = ne*KbT*scaling
    r_set = np.logspace(np.log10(rmin),np.log10(rvir),num=101)
    V = 4.0/3*np.pi*r_set[-1]**3
    return np.sqrt(2.0*Uth*V/tools.Integrate(equipartition_bfield_profile(r_set,rmin,w)**2*r_set**2,r_set))*1e6