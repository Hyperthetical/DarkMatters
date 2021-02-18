#cython: language_level=3
import numpy as np
from scipy.integrate import simps as integrate
from scipy.special import sici,erfc
from scipy.interpolate import interp1d
try:
    from scipy.special import sph_jn
except:
    from scipy.special import spherical_jn as sph_jn


def deltaHsq(k,ns,kst):
    d0sq = 4.58e10*np.exp(2.48*(ns-1))
    alphas = alpha(k,ns)
    return d0sq*(k/kst)**(ns -1 + alphas*np.log(k/kst))

def window(x):
    #top-hat window function in fourier space
    return 3*(np.sin(x) - x*np.cos(x))/x**3

def alpha(k,ns):
    return -0.0068 #1512.04597

def ns(k):
    return 0.9652 #1512.04597

def Ci(x):
    ye = 0.5772156649 #Euler-Mascheroni constant
    return x**(-2)*(sici(x)[1] - np.log(x) - ye)

def transfer(x):
    ye = 0.5772156649 #Euler-Mascheroni constant
    xr = 1.0/np.sqrt(3.0)
    Tx = np.zeros(len(x))
    Tr = 3/xr*sph_jn(1,xr)[0][1]*np.ones(len(x))
    for i in range(0,len(x)):
        Tx[i] = 6/x[i]**2*(np.log(x[i])+ye-0.5-Ci(x[i])+0.5*sph_jn(0,x[i])[0][0])
    return Tx/Tr

def specAmp(n,k,kst):
    x = np.logspace(-5,10,num=100)
    alphas = alpha(k,n)
    intg = x**(n+2+alphas*np.log(k*x/kst))*(k/kst)**(alphas*np.log(x))
    intg = intg*window(x)**2*transfer(x)**2
    return integrate(intg,x)

def glinear(z,**cosmo):
    om = omegaZ(cosmo['omega_M_0'],z,**cosmo)
    om0 = cosmo['omega_M_0']
    ol = 1.0 - om
    ol0 = 1.0 - om0
    N = 2.5*om0*(om0**(4.0/7) - ol0 + (1 + 0.5*om0)*(1 + 1.0/70*ol0))**(-1)
    g = 2.5*om*(om**(4.0/7) - ol + (1 + 0.5*om)*(1 + 1.0/70*ol))**(-1)/(1 + z)/N
    return g

#P&S mass function for UCMHs derived like Vittorio & Colafrancesco with tweaks
#M is the halo mass, n is the power-law spectral index, n <= 1.25 from current limits for UCMH
def dndmUCMH(M,z,n,**cosmo):
    rhobar = rho_crit(0.0,**cosmo)*cosmo['omega_M_0']/1e15 #background density at z = 0.0
    j = 2
    N0 = 4.7e-4*cosmo['h']**3 #using Mstar = M_8
    y = 2-(n+3)/6.0 #exponent from differentiation of sigma
    r8 = 8/cosmo['h'] #radius for M_8 and sigma_8
    b = 1.0 #sigma_8 normalisation
    dc = 1.0e-3 #UCMH minimum collapse contrast
    Mstar = 4.0/3.0*np.pi*(r8)**3*rhobar #reference mass
    dz = glinear(z,**cosmo) #linear growth function
    return N0*j/np.sqrt(2*np.pi)*(n+3)/6.0*dc*b/Mstar*(M/Mstar)**(-y)/dz*np.exp(-0.5*dc**2*b**2/dz**2*(M/Mstar)**((n+3)/3.0))


def sigmaHsq(R):
    k = 1.0/R
    n = ns(k)
    kst = 0.05 #Mpc^-1
    return specAmp(n,k,kst)*deltaHsq(k,n,kst)

def tzc(z,h):
    return 2.0/3/h/1e2/(1+z)**(1.5)

def omega_m(z,**cosmo):
    return 1.0/(1.0 + (1.0-cosmo['omega_M_0'])/cosmo['omega_M_0']*(1+z)**(-3))

def rho_crit(z,**cosmo):
    return 1e9*2.7755e-2*hubble_z(z,**cosmo)**2

def deltaMin(k,t,zc,**cosmo):
    ye = 0.5772156649 #Euler-Mascheroni constant
    x = 1.0/np.sqrt(3.0)
    om_m = 0.315
    H0 = 67.3
    om_r = 0.5e-3
    Hz = hubble_z(zc,**cosmo)
    kappa = k*np.sqrt(om_r)/H0/om_m
    Tx = 6/x**2*(np.log(x)+ye-0.5-Ci(x)+0.5*sph_jn(0,x)[0][0])
    return 2.0/9*(3*np.pi/2)**(2.0/3)*Tx/k**2/tau(kappa)*(k/Hz)**2

def tau(x):
    t = np.log(1 + (0.124*x)**2)/(0.124*x)**2
    t = t*np.sqrt((1 + (1.257*x)**2 + (0.4452*x)**4 + (0.2197*x)**6)/(1 + (1.606*x)**2 + (0.8568*x)**4 + (0.3927*x)**6))
    return t

def sigmaApprox(phase,**cosmo):
    sig = 9.5e-5*(horizonMassT(phase,**cosmo)*0.5e-33)**(0.25*(1-cosmo['n']))
    return sig

def omegaUCMH(z,**cosmo):
    sig = sigmaApprox(z,**cosmo)
    deltamin = 1e-3
    deltamax = 1.0/3
    deltax = np.logspace(np.log10(deltamin),np.log10(deltamax),num=200)
    return 1.0/np.sqrt(2*np.pi)/sig*integrate(np.exp(-deltax**2/(2*sig**2)),deltax)

def collapseFrac(R):
    sigmasq = sigmaHsq(R)
    deltamin = 1e-3
    deltamax = 1.0/3
    deltax = np.logspace(np.log10(deltamin),np.log10(deltamax),num=200)
    return 1.0/np.sqrt(2*np.pi*sigmasq)*integrate(np.exp(-deltax**2/(2*sigmasq)),deltax)

#number of UCMHs in a structure with mass = targetMass
def numUCMH(targetMass,phase,**cosmo):
    fx = cosmo['omega_dm_0']/cosmo['omega_M_0']
    f = fx*ucmhFrac(phase,**cosmo)
    return f*targetMass/fx/massUCMH(0,phase,**cosmo)


def horizonMassT(phase,**cosmo):
    mhteq = 6.5e15*(cosmo['omega_M_0']*cosmo['h']**2)**(-2) #solar masses
    Teq = 5.5*cosmo['omega_M_0']*cosmo['h']**2 #eV
    geq = 3.91
    gT = cosmo[phase][1]
    T = cosmo[phase][0]
    g = (geq/gT)**(1.0/3)
    t = Teq/T
    return mhteq*g**2*t**2

def smoothR(phase,**cosmo):
    mH = horizonMassT(phase,**cosmo)
    mHeq = horizonMassZeq(**cosmo)
    keq = getKeq(**cosmo)
    gi = 100.0
    geq = 3.0
    return np.sqrt(mH/mHeq*(gi/geq)**(1.0/3))/keq

def massUCMH(z,phase,**cosmo):
    fx = cosmo['omega_dm_0']/cosmo['omega_M_0']
    dm = fx*horizonMassT(phase,**cosmo)*(1+getZeq(**cosmo))/(1+cosmo[phase][2])
    if z < 10:
        zn = 10
    else:
        zn = z
    return dm*(1+getZeq(**cosmo))/(1+zn)

def horizonMassZeq(**cosmo):
    return 6.5e15/(cosmo['omega_M_0']*cosmo['h']**2)**2

def rHalo(z,phase,**cosmo):
    mh = massUCMH(z,phase,**cosmo)
    return 0.019*1e3/(1+z)*mh**(1.0/3)*1e-6

def rConvert(z,mh,**cosmo):
    return 0.019*1e3/(1+z)*mh**(1.0/3)*1e-6


def deltaM(phase,**cosmo):
    mh = horizonMassT(phase,**cosmo)
    fx = cosmo['omega_dm_0']/cosmo['omega_M_0']
    return fx*(1+getZeq(**cosmo))/(1+cosmo[phase][2])*mh

def ucmhFrac(phase,**cosmo):
    #R = mass_to_radius(M0,**cosmo)
    R = smoothR(phase,**cosmo)
    M0 = deltaM(phase,**cosmo)
    M = massUCMH(0,phase,**cosmo)
    beta = collapseFrac(R)
    return M/M0*beta

def omegaZ(z,omega,**cosmo):
    return 1.0/(1.0 + (1.0-omega)/omega*(1+z)**(-3))

def age_flat(z,**cosmo):
    om = cosmo['omega_M_0'] 
    H100_s = 3.24077648681e-18
    lam = 1. - cosmo['omega_M_0'] 
    t_z = (2.*arcsinh(sqrt(lam/om) * (1. + z)**(-3./2.)) / (H100_s * cosmo['h'] * 3. * sqrt(lam))) 
    return t_z 

def rhoUCMH(r,z,mh,sigV,mx,decay=False,**cosmo):
    fx = cosmo['omega_dm_0']/cosmo['omega_M_0']
    rh = rConvert(z,mh,**cosmo)
    tz = age_flat(z,**cosmo) #s
    teq = age_flat(10.0,**cosmo) #s
    GyrToS = 1e9*365.25*24*3600
    #print tz/GyrToS, 59e-3
    dt = (0.49 - 77e-3)*1e9*365.25*24*3600  #time between zeq and z=0 in s
    dt = (13.76 - 59e-3)*1e9*365.25*24*3600  #time between zeq and z=0 in s
    dt = (tz - 59e-3*GyrToS)  #time between zeq 1110.2484 and z=0 in s
    rhomax = mx/sigV/dt*1.602e-40*0.5/(3e8)**2*(3.09e24)**3
    rho = 3*fx*mh/(16*np.pi*rh**(0.75)*r**(9.0/4))
    rmin = 2.9e-7*rConvert(z,mh,**cosmo)*mh**(-0.06)*(1e3/(getZeq(**cosmo)+1))**2.43
    rhormin = 3*fx*mh/(16*np.pi*rh**(0.75)*rmin**(9.0/4))
    if np.where(rho > rhomax) != np.array([]):
        rho = np.where(rho > rhomax, rhomax, rho)
    else:
        rho = np.where(r <= rmin,rhormin,rho)
    return rho

def rhoUCMHMoore(r,z,mh,sigV,mx,**cosmo):
    tz = age_flat(z,**cosmo) #s
    teq = age_flat(10.0,**cosmo) #s
    GyrToS = 1e9*365.25*24*3600
    #print tz/GyrToS, 59e-3
    dt = (0.49 - 77e-3)*1e9*365.25*24*3600  #time between zeq and z=0 in s
    dt = (13.76 - 59e-3)*1e9*365.25*24*3600  #time between zeq and z=0 in s
    dt = (tz - 59e-3*GyrToS)  #time between zeq 1110.2484 and z=0 in s
    rhomax = mx/sigV/dt*1.602e-40*0.5/(3e8)**2*(3.09e24)**3
    zc = 1e3;ks=6.8 #kpc-1
    rhos = 30*(1+zc)**3*rho_crit(z,**cosmo) #Msol Mpc^-3
    rs = 0.7/((1+zc)*ks*1e3) #Mpc
    rho = rhos/(r/rs)**1.5/(1+r/rs)**1.5
    #print(max(rho))
    rset = np.logspace(np.log10(rs*1e-7),np.log10(rConvert(z,mh,**cosmo)),num=200)
    rhoset = rhos/(rset/rs)**1.5/(1+rset/rs)**1.5
    norm = mh/integrate(rhoset*4*np.pi*rset**2,rset)
    rho = rho*norm
    rho = np.where(rho > rhomax, rhomax, rho)
    #print(max(rho))
    #print(integrate(rho*4*np.pi*r**2,r)/mh)
    return rho

def rhoDecayUCMH(r,z,mh,G,mx,**cosmo):
    fx = cosmo['omega_dm_0']/cosmo['omega_M_0']
    rh = rConvert(z,mh,**cosmo)
    dt = (0.49 - 77e-3)*1e9*365.25*24*3600  #time between zeq and z=0 in s
    rhobar = mh/(4/3.0*rh**3)
    rhodecaybar = rhobar  - mx*G*dt/(4.0/3*rh**3)

    rho = 3*fx*mh/(16*np.pi*rh**(0.75)*r**(9.0/4))
    print(rhodecaybar/rhobar)

def rhoNFW(r,rc,rhos,**cosmo):
    #draco data as calcuated at z = 0
    #rc = 0.000429676695406
    #rhos = 241969.9026*rho_crit(0,**cosmo)
    rho = 1.0/((r/rc)*(1.0+r/rc)**2)*rhos
    return rho

def rMin(z,mh):
    return 5.1e-7*(1e3/(z+1))**2.43*mh**0.27*1e-6 #Mpc

def anniUCMH(z,phase,sigV,mx,**cosmo):
    rh = rHalo(z,phase,**cosmo)
    r = np.logspace(np.log10(rh*1e-6),np.log10(rh),num=100)
    rho = rhoUCMH(r,z,phase,sigV,mx,**cosmo)
    return 4*np.pi*integrate(rho**2*r**2,r)

def hubble_z(z,**cosmo):
    H0 = 100
    return H0*cosmo['h']*np.sqrt(cosmo['omega_M_0']*(1.0+z)**3+cosmo['omega_k_0']*(1+z)**2+cosmo['omega_lambda_0'])

def getZeq(**cosmo):
    return 2.32e4*cosmo['omega_M_0']*cosmo['h']**2 - 1

def getKeq(**cosmo):
    return 0.07*cosmo['omega_M_0']*cosmo['h']**2

def testUCMH():
    #phasedict = {'EW':np.array([2e11,107,1e15]),'QCD':np.array([2e8,55,1e12]),'EE':np.array([0.51e6,10.8,2e9])}
    cosmo = {'omega_M_0':0.3089, 'omega_lambda_0':0.6911, 'omega_k_0':0.0, 'h':0.6774, 'omega_dm_0':0.2589, 'omega_b_0':0.0486, 'sigma_8':0.897, 'n':1.25, 'omega_n_0':0.0, 'N_nu':0, 'EW':np.array([2e11,107,1e15]),'QCD':np.array([2e8,55,1e12]),'EE':np.array([0.51e6,10.8,2e9])}
    k = np.logspace(1,8,num=100)
    keq = 0.07*cosmo['omega_M_0']*cosmo['h']**2

    rhoDecayUCMH(1.0,0.0,1e3,1e-26,1e3,**cosmo)


    zset = np.logspace(3,10,num=100)
    fx = cosmo['omega_dm_0']/cosmo['omega_M_0']
    T = 2e11; gT = 107;zx = 1e15
    T = 0.51e6; gT = 10.8;zx = 2e9
    T = 200e6; gT = 55;zx = 1e12

#testUCMH()
