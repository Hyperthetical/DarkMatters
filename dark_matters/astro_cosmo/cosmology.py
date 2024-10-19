"""
DarkMatters.astro_cosmo module for calculating cosmology dependent functions
"""
from scipy.integrate import quad
from scipy.optimize import newton
from scipy.integrate import simpson as integrate
import numpy as np

def rho_crit(z,cosmo):
    """
    Calculates universe critical density at z

    Arguments
    ---------------------------
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    rho_crit : float
        Critical density [Msun/Mpc^3]
    """
    return 1e9*2.7755e-2/hubble_func(z,cosmo)**2

def omega_m(z,cosmo):
    """
    Calculates matter fraction at z

    Arguments
    ---------------------------
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    omega_m : float
        Matter fraction at z
    """
    return 1.0/(1.0 + (1.0-cosmo['omega_m'])/cosmo['omega_m']/(1+z)**(3))

def delta_c(z,cosmo):
    """
    Calculates virial density contrast z

    Arguments
    ---------------------------
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    delta_c : float
        Virial density contrast at z
    """
    x = 1.0 - omega_m(z,cosmo)
    return (18.0*np.pi**2 - 82.0*x - 39.0*x**2)#/omega_m(z,cosmo)

def rvir_from_mvir(mvir,z,cosmo):
    """
    Virial radius from virial mass

    Arguments
    ---------------------------
    mvir : float
        Virial mass [Msun]
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    rvir : float
        Virial radius [Mpc]
    """
    return (0.75*mvir/(np.pi*delta_c(z,cosmo)*rho_crit(z,cosmo)))**(1.0/3.0) #in Mpc

def mvir_from_rvir(rvir,z,cosmo):
    """
    Virial mass from virial radius

    Arguments
    ---------------------------
    rvir : float
        Virial radius [Mpc]
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    mvir : float
        Virial mass [Msun]
    """
    return 4*np.pi/3.0*delta_c(z,cosmo)*rho_crit(z,cosmo)*rvir**3

def halo_scale(M,z,cosmo):
    """
    Halo scale radius

    Arguments
    ---------------------------
    M : float
        Virial mass [Msun]
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    rs : float
        Scale radius [Mpc]
    """
    return rvir_from_mvir(M,z,cosmo)/cvir(M,z,cosmo)

def rho_nfw_norm_relative(cv,z,cosmo):
    """
    Rhos relative to rho_crit for NFW halo

    Arguments
    ---------------------------
    cv : float
        Virial concentration
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    rho_normRelative : float
        Rhos/rho_crit
    """
    return delta_c(z,cosmo)/3.0*cv**3/(np.log(cv+1.0) - cv/(1.0+cv))

def hubble_func(z,cosmo):
    """
    1/H(z)

    Arguments
    ---------------------------
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    1/H(z) : float
        Inverse Hubble parameter [(Mpc s)/km]
    """
    H0 = 100.0
    w_k = 1 - cosmo['omega_m'] - cosmo['omega_l']
    return (H0*cosmo['h']*np.sqrt(cosmo['omega_m']*(1.0+z)**3+w_k*(1+z)**2+cosmo['omega_l']))**(-1)

def dist_co_move(z,cosmo):
    """
    Co-moving distance to z

    Arguments
    ---------------------------
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    dcm : float
        Co-moving distance to z [Mpc]
    """
    c = 2.99792458e5 #km s^-1
    w_k = 1 - cosmo['omega_m'] - cosmo['omega_l']
    dc = quad(hubble_func,0,z,args=(cosmo))[0]*c
    dh = c/(100*cosmo['h'])
    if(w_k == 0.0):
        dcm = dc
    elif(w_k > 0.0):
        dcm = dh/np.sqrt(w_k)*np.sinh(np.sqrt(w_k)*dc/dh)
    elif(w_k < 0.0):
        dcm = dh/np.sqrt(-w_k)*np.sinh(np.sqrt(-w_k)*dc/dh)
    return dcm

def dist_luminosity(z,cosmo):
    """
    Luminosity distance to z

    Arguments
    ---------------------------
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    dl : float
        Luminosity distance [Mpc]
    """
    return (1.0+z)*dist_co_move(z,cosmo)

def dist_angular(z,cosmo):
    """
    Angular diameter distance to z

    Arguments
    ---------------------------
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    da : float
        Angular diameter distance [Mpc]
    """
    return dist_co_move(z,cosmo)/(1.0+z)




def cvir(Mvir,z,cosmo):
    """
    Virial concentration function selector

    Arguments
    ---------------------------
    Mvir : float
        Virial mass [Msun]
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    cvir : float
        Virial concentration
    """
    if cosmo['cvir_mode'] == "p12":
        cvir_func = cvir_p12
    elif cosmo['cvir_mode'] == "munoz_2011":
        cvir_func = cvir_munoz
    elif cosmo['cvir_mode'] == "bullock_2001":
        cvir_func = cvir_bullock2001
    elif cosmo['cvir_mode'] == "cpu_2006":
        cvir_func = cvir_cpu

    return cvir_func(Mvir,z,cosmo)

def cvir_bullock2001(M,z,cosmo):
    """
    Calculates Virial concentration given (M,z) (Bullock 2001)

    Arguments
    ---------------------------
    M : float
        Virial mass [Msun]
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    cvir : float
        Virial concentration
    """
    return 9.0/(1.0+z)*(M/1.3e13*cosmo['h'])**(-0.13)

def cvir_p12_param(M,z,cosmo):
    """
    Parametric Mass-concentration relation from Sanchez-Conde & Prada 2013

    Arguments
    ---------------------------
    M : float
        Virial mass [Msun]
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    cvir : float
        Virial concentration
    """
    c = np.array([37.5153,-1.5093,1.636e-2,3.66e-4,-2.89237e-5,5.32e-7])
    cv = 0.0
    for i in range(0,len(c)):
        cv += c[i]*np.log(M*cosmo['h'])**i
    return c200_to_cvir(cv,z,cosmo)

def c200_to_cvir(c200,z,cosmo):
    """
    Fitting function from 1005.0411, valid for NFW

    Arguments
    ---------------------------
    c200 : float
        c200 concentration
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    cvir : float
        Virial concentration
    """
    dc = delta_c(z,cosmo)
    a = -1.119*np.log10(dc) + 3.537
    b = -0.967*np.log10(dc) + 2.181
    return a*c200 + b

def cvir_p12(M,z,cosmo):
    """
    Mass-concentration relation from Sanchez-Conde & Prada 2013

    Arguments
    ---------------------------
    M : float
        Virial mass [Msun]
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    cvir : float
        Virial concentration
    """
    def sigma_param(M,z,cosmo):
        """
        Overdensity sigma from Sanchez-Conde & Prada 2013

        Arguments
        ---------------------------
        M : float
            Virial mass [Msun]
        z : float
            Redshift
        cosmo : dictionary
            Cosmology parameter

        Returns
        ---------------------------
        sigma : float
            Relative mass overdensity
        """
        y = (M*cosmo['h']/1.0e12)**(-1)
        return glinear(z,cosmo)*16.9*y**(0.41)/(1+1.102*y**(0.2)+6.22*y**(0.333))

    def smin(x,y):
        """
        Smin function from Sanchez-Conde & Prada 2013

        Arguments
        ---------------------------
        x : float
            Scale-factor at ML equality times a(z)
        y : float
            Scale-factor at ML equality

        Returns
        ---------------------------
        Smin : float
            Fitting ratio for cvir_p12
        """
        s0 = 1.047;s1 = 1.646;beta = 7.386;x1 = 0.526 #fitting parameters
        return (s0 + (s1-s0)*(np.arctan(beta*(x-x1))/np.pi+0.5))/(s0 + (s1-s0)*(np.arctan(beta*(y-x1))/np.pi+0.5))

    def cmin(x,y):
        """
        Cmin function from Sanchez-Conde & Prada 2013

        Arguments
        ---------------------------
        x : float
            Scale-factor at ML equality times a(z)
        y : float
            Scale-factor at ML equality

        Returns
        ---------------------------
        Cmin : float
            Fitting ratio for cvir_p12
        """
        c0 = 3.681;c1 = 5.033;alpha = 6.948;x0 = 0.424 #fitting parameters
        return (c0 + (c1-c0)*(np.arctan(alpha*(x-x0))/np.pi+0.5))/(c0 + (c1-c0)*(np.arctan(alpha*(y-x0))/np.pi+0.5))
    a = 1.0/(1+z)
    x = (cosmo['omega_l']/cosmo['omega_m'])**(1.0/3)*a
    y = x/a #this ensures b1 and b0 are unity at z = 0
    b0 = cmin(x,y)
    b1 = smin(x,y)
    sigma_p = b1*sigma_param(M,z,cosmo)
    A = 2.881;b=1.257;c=1.022;d=0.060
    csig = A*((sigma_p/b)**c + 1)*np.exp(d/sigma_p**2)
    c200 = b0*csig
    return c200_to_cvir(c200,z,cosmo)

def cvir_cpu(Mvir,z,cosmo):
    """
    Virial concentration from Colafrancesco, Profumo, & Ullio 2006

    Arguments
    ---------------------------
    Mvir : float
        Virial mass [Msun]
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    cvir : float
        Virial concentration
    """
    def find_zc(z,Mvir,cosmo):
        m = Mvir*0.015
        r8 = 8/cosmo['h']
        z0 = 0.0
        rcut = (3*1.0e-6/(4*np.pi*omega_m(z0,cosmo)*rho_crit(z0,cosmo)))**(1.0/3)
        r = (3*m/(4*np.pi*omega_m(z0,cosmo)*rho_crit(z0,cosmo)))**(1.0/3)
        sig8 = sigma_l(r8,0,rcut,rcut,z0,cosmo)/0.897**2
        sig = np.sqrt(sigma_l(r,0,rcut,rcut,z0,cosmo)/sig8)
        return glinear(z,cosmo)*sig - 1.686

    zc = newton(find_zc,1,args=(Mvir,cosmo))
    return (delta_c(zc,cosmo)*omega_m(z,cosmo)/delta_c(z,cosmo)/omega_m(zc,cosmo))**(1.0/3)*(1+zc)/(1+z)

def cvir_munoz(Mvir,z,cosmo):
    """
    Halo concentration from Munoz 2010

    Arguments
    ---------------------------
    Mvir : float
        Virial mass [Msun]
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    cvir : float
        Virial concentration
    """
    w = 0.029
    m = 0.097
    alpha = -110.001
    beta = 2469.72
    gamma = 16.885
    a = w*z - m
    b = alpha/(z+gamma) + beta/(z+gamma)**2
    logc = a*np.log10(Mvir*cosmo['h']) + b
    return 10**(logc)

def sigma_l(r,l,rc,rmin,z,cosmo):
    """
    Average relative excess mass within sphere of radius r

    Arguments
    ---------------------------
    r : float
        Radius of sphere [Mpc]
    l : integer
        Distribution moment
    rc : float
        Cut-off radius [Mpc]
    rmin : float
        Minimum considered radius [Mpc]
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    sigma_l : float
        Average relative excess mass in radius r
    """
    def pspec(k,z,cosmo):
        """
        Matter perturbation power spectrum

        Arguments
        ---------------------------
        M : float
            Virial mass [Msun]
        z : float
            Redshift
        cosmo : dictionary
            Cosmology parameter

        Returns
        ---------------------------
        dl : float
            Luminosity distance [Mpc]
        """
        q = k/(omega_m(z,cosmo)*cosmo['h']**2)
        return k*(np.log(1+2.34*q)/(2.34*q)/(1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3+(6.71*q)**4)**0.25)**2

    def window(x):
        """
        Top-hat window function in fourier space

        Arguments
        ---------------------------
        x : float
            Variable

        Returns
        ---------------------------
        window : float
            Fourier transform of top-hat function
        """
        return 3*(np.sin(x) - x*np.cos(x))/x**3
    n = 101
    kmax = 1.0/rmin
    kcut = 1.0/rc
    kmin = 1.0e-8*kmax
    kset = np.zeros(n,dtype=float)
    kint = np.zeros(n,dtype=float)
    kset = np.logspace(np.log10(kmin),np.log10(kmax),num=n)
    kint = 0.5*kset**(2*(1+l))*pspec(kset,z,cosmo)*window(kset*r)**2*np.exp(-kset/kcut)/np.pi**2
    return integrate(y=kint,x=kset)

def glinear(z,cosmo):
    """
    Linear growth function at z

    Arguments
    ---------------------------
    z : float
        Redshift
    cosmo : dictionary
        Cosmology parameter

    Returns
    ---------------------------
    glinear : float
        Growth function g(z)
    """
    g = 1.0
    if(cosmo['omega_l'] == 0.0 and cosmo['omega_m'] == 1.0):
        g = (1+z)**(-1)
    elif(cosmo['omega_m'] < 1.0 and cosmo['omega_l'] == 0.0):
        x0 = (1.0/cosmo['omega_m'] - 1)
        x = x0/(1+z)
        Ax = 1 + 3/x + 3*np.sqrt(1+x)/x**1.5*np.log(np.sqrt(1+x)-np.sqrt(x))
        Ax0 = 1 + 3/x0 + 3*np.sqrt(1+x0)/x0**1.5*np.log(np.sqrt(1+x0)-np.sqrt(x0))
        g = Ax/Ax0
    elif(cosmo['omega_l'] > 0 and cosmo['omega_l'] == 1.0 - cosmo['omega_m']):
        om = omega_m(z,cosmo)
        om0 = omega_m(0.0,cosmo)
        ol = 1.0 - om
        ol0 = 1.0 - om0
        #this approximation very closely matches true expression commented out below
        N = 2.5*om0*(om0**(4.0/7) - ol0 + (1 + 0.5*om0)*(1 + 1.0/70*ol0))**(-1)
        g = 2.5*om*(om**(4.0/7) - ol + (1 + 0.5*om)*(1 + 1.0/70*ol))**(-1)/(1 + z)/N
        #x0 = (2*(1.0/w_m - 1))**(1.0/3)
        #x = x0/(1+z)
        #xset = linspace(0,x,num=101)
        #x0set = linspace(0,x0,num=101)
        #aset = (xset/(xset**3+2))**1.5
        #a0set = (x0set/(x0set**3+2))**1.5
        #Ax = np.sqrt(x**3 + 2)/x**1.5*integrate(aset,xset)
        #Ax0 = np.sqrt(x0**3 + 2)/x0**1.5*integrate(a0set,x0set)
        #ga = Ax/Ax0
    return g