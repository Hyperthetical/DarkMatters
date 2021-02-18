#cython: language_level=3
from numpy import *
try:
    from wimp_tools import cosmology,tools,environments,astrophysics
except:
    import wimp_tools.cosmology as cosmology
    import wimp_tools.tools as tools
    import wimp_tools.environments as environments
    import wimp_tools.astrophysics as astrophysics

def b_sanchez(M,Mmin,z,cosmo):
    alpha = -2.0
    A = 0.012
    mset = logspace(log10(Mmin),log10(M),num=50)
    c = cosmology.cvir_p12(mset,z,cosmo)
    cM = cosmology.cvir_p12(M,z,cosmo)
    f = log(1+c) - 1.0/(1+c)
    fM = log(1+cM)-1.0/(1+cM)
    L = 4.0*pi*mset*c**3/f**2
    LM = 4.0*pi*M*cM**3/fM**2
    dNdm = A/M*(mset/M)**alpha
    integ = L*2*dNdm
    return tools.Integrate(integ,mset)/LM

def b_p12(M,cosmo):
    bi = array([-0.442,0.0796,-0.0025,4.77e-6,4.77e-6,-9.69e-8])
    lm = log(M)
    n = array([0,1,2,3,4,5],dtype=int)
    bp = (bi*lm**n).sum()
    return 10**bp

def rho_sub_dist(r_set,rho,rb,N,dmmod,alphas):
    #returns spacial density of sub halos relative to rho_crit
    #r_set -> position sampling
    #rc -> scale radius
    #rb -> baising radius
    #dmmod -> dark matter halo profile model index
    #alpha is the Einasto parameter
    if(dmmod != -1):
        n = len(r_set)
        rho_sub = zeros(n,dtype=float)
        for i in range(0,n):
            r = r_set[i]
            rho_sub[i] = rho[i]*(r/rb)/(1+r/rb)
    else:
        rho_sub = astrophysics.rho_dm(r_set,rb,dmmod,alphas)*N 
    return rho_sub

def radio_power_ratio(halo,phys,sim):
    n = sim.n #number of r shells
    k = len(phys.spectrum[0]) #number of E bins
    num = sim.num  #number of frequency sampling points
    ntheta = 100   #angular integration points

    emm = zeros((num,n),dtype=float)   #emmisivity
    theta_set = zeros(ntheta,dtype=float) #angular sampling values
    tset = zeros(ntheta,dtype=float) #angular sampling values
    theta_int = zeros(ntheta,dtype=float) #angular integral sampling
    int_1 = zeros(k,dtype=float) #energy integral sampling

    r0 = 2.82e-13  #electron radius (cm)
    me = 0.511e-3  #electron mass (GeV)
    c = 3.0e10     #speed of light (cm s^-1)

    theta_set = linspace(1e-2,pi,num=ntheta)  #choose angles 0 -> pi
    nu_cut = 1e12 #MHz -> cut-off to stop synchrotron calculations works up 3 TeV m_x
    av = zeros(n)
    for i in range(0,num):  #loop over freq
        nu = halo.f_sample[i]*(1+halo.z)
        P_S = zeros(n)
        if nu > nu_cut*(1+halo.z):
            emm[i,:] = zeros(n)[:]
        else:
            P_Sj = zeros((k,n))
            r = halo.r_sample[0]
            bmu = halo.b_sample
            ne = halo.ne_sample
            nu0 = 2.8*bmu*1e-6
            nup = 8980.0*sqrt(ne)*1e-6
            a = 2.0*pi*sqrt(3.0)*r0*me/c*1e6*nu0  #gyro radius

            for l in range(0,k):   #loop over energy
                g = phys.spectrum[0][l]
                x = 2.0*nu/(3.0*nu0*g**2)*(1+(g*nup/nu)**2)**1.5 #dimensionless integration
                xmatrix = tensordot(x,1.0/sin(theta_set),axes=0)
                theta_int = 0.5*sin(theta_set)**2*tools.int_bessel(xmatrix)  #theta integrand vectorised
                #integrate over that and factor in electron densities
                P_Sj[l] = a*tools.Integrate(theta_int,theta_set)
            for j in range(0,n):
                P_S[j] = tools.Integrate(P_Sj[:,j],phys.spectrum[0])
            av += P_S/P_S[0]/num
    return av




def mass_sub(ms,mstar,mu):
    #this is the un-normalised sub halo mass distribution function
    #ms is the sub halo mass
    #mstar is the mass scale
    #mu is the exponent
    return (ms/mstar)**mu

def mass_sub_s(ms,mcut,mu):
    return ms**(mu)*exp(-(ms/mcut)**(-2.0/3))

def prob_sub_cs(cs,sigc,csbar):
    #this is log-normal distribution for sub halo concentrations
    #cs is the sub halo concentration parameter
    #sigc is the log normal deviation of cs
    #csbar is the mean of cs
    return 1.0/(sqrt(2.0*pi)*sigc*cs)*exp(-0.5*(log10(cs)-log10(csbar))**2/sigc**2)

def dns_sub(mass_set,mstar,mu):
    n = len(mass_set)
    dns = zeros(n,dtype=float)
    for i in range(0,n):
        dns[i] = mass_sub_s(mass_set[i],mstar,mu)
    return dns

def rhobars(Mvir,Rvir,rc,dmmod,alpha):
    I1 = astrophysics.rho_volume_int(Rvir/rc,1.0,1,dmmod,alpha)
    return Mvir*I1/rc**3

def delta(Mvir,fs,z,cos_env,dmmod,alpha):
    mnum = 51
    cnum = 51
    if(Mvir > 1e15):
        mmax = 1e15
    else:
        mmax = Mvir
    mmin = 1.0e-6
    sigc = 0.14  
    norm = 0.0
    F = 1.5

    mass = zeros(mnum,dtype=float)
    cs = zeros(cnum,dtype=float)
    int_c = zeros(cnum,dtype=float)
    prob = zeros(cnum,dtype=float)
    deltams = zeros(mnum,dtype=float)

    for i in range(0,mnum):
        mass[i] = 10.0**(log10(mmin) + (log10(mmax) - log10(mmin))*i/(mnum-1))
    dns = dns_sub(mass,mmin,-1.9)
    deltams = dns*mass
    Nm = fs*Mvir/tools.Integrate(deltams,mass)
    deltams = zeros(mnum,dtype=float)
    for i in range(0,mnum):
        cbar = F*cosmology.cvir_cpu(mass[i],z,cos_env)
        cmax = 10.0**(log10(cbar) + 3*sigc)
        cmin = 10.0**(log10(cbar) - 4*sigc)
        for j in range(0,cnum):
            cs[j] = cmin + (cmax - cmin)*j/(cnum-1)
            prob[j] = prob_sub_cs(cs[j],sigc,cbar)
        norm = norm + tools.Integrate(prob,cs)
        for j in range(0,cnum):
            I1 = 1.0/astrophysics.rho_volume_int(cs[j],1.0,1,dmmod,alpha)/4.0/pi
            I2 = 1.0/astrophysics.rho_volume_int(cs[j],1.0,2,dmmod,alpha)/4.0/pi
            int_c[j] = prob[j]*cs[j]**3*I2/I1**2*cosmology.delta_c(z,cos_env)/cosmology.omega_m(z,cos_env)/3.0
        deltams[i] = tools.Integrate(int_c,cs)*dns[i]*mass[i]*Nm/(Mvir*fs)
    return tools.Integrate(deltams,mass)
    
def rb_norm_nfw(rb,rc,rho_s,rv,msm):
    #returns how close rb is to correct value for given msm (0 means rb is exact)
    #rb -> biasing radius
    #rc -> scaling radius
    #rho_s -> density parameter of halo
    #rv -> halo virial radius
    #msm is the smooth mass fraction of the halo
    a = rv*(rc - rb) + rb*(rv + rc)*log(rb/rc*(rv + rc)/(rb + rv))
    b = msm*(rb - rc)**2*(rv + rc)
    return 4.0*pi*rb*rc**3*rho_s*a/b - 1.0

def rb_norm(rb,rc,rho_s,r_set,msm,dmmod,alpha):
    #returns how close rb is to correct value for given msm (0 means rb is exact)
    #rb -> biasing radius
    #rc -> scaling radius
    #rho_s -> density parameter of halo
    #r_set -> sampled positions
    #msm is the smooth mass fraction of the halo
    #dmmod -> dark matter halo profile model index
    #alpha is the Einasto parameter
    n = len(r_set)
    int_set = zeros(n,dtype=float)
    rho = astrophysics.rho_dm(r_set,rc,dmmod,alpha)
    int_set = 4.0*pi*r_set**2*rho*rho_s/msm/(1+r_set/rb)
    return tools.Integrate(int_set,r_set) - 1.0

def find_rb(rc,rho_s,rv,msm,dmmod,alpha):
    #this finds the correct value of rb for given halo
    #the algorithm is a bisection algorithm
    #rc -> scaling radius
    #rho_s -> density parameter of halo
    #msm is the smooth mass fraction of the halo
    #dmmod -> dark matter halo profile model index
    #alpha is the Einasto parameter
    n = 101
    rmin = rc*1e-3
    r_set = zeros(n,dtype=float)
    for i in range(0,n):
        r_set[i] = 10.0**(log10(rmin) + (log10(rv) - log10(rmin))*i/(n-1))
    rbmax = 100.0*rc
    rbmin = 0.0001*rc
    rbmid = 0.5*(rbmax + rbmin)
    found = False
    rbf = 10.0*rc
    if(dmmod != 1):
        fmax = rb_norm(rbmax,rc,rho_s,r_set,msm,dmmod,alpha)
        fmid = rb_norm(rbmid,rc,rho_s,r_set,msm,dmmod,alpha)
        fmin = rb_norm(rbmin,rc,rho_s,r_set,msm,dmmod,alpha)
    else:
        fmax = rb_norm_nfw(rbmax,rc,rho_s,rv,msm)
        fmid = rb_norm_nfw(rbmid,rc,rho_s,rv,msm)
        fmin = rb_norm_nfw(rbmin,rc,rho_s,rv,msm)
    if(abs(fmax) <= 1e-5):
        found = True 
        rbf = rbmax
    elif(abs(fmin) <= 1e-5):
        found = True 
        rbf = rbmin
    elif(abs(fmid) <= 1e-5):
        found = True 
        rbf = rbmid
    n = 0
    while((not found) and (n < 50)):
        if(fmid/fmax < 0.0):
            rbmin = rbmid
        else:
            rbmax = rbmid
        rbmid = 0.5*(rbmin + rbmax)
        if(dmmod != 1):
            fmax = rb_norm(rbmax,rc,rho_s,r_set,msm,dmmod,alpha)
            fmid = rb_norm(rbmid,rc,rho_s,r_set,msm,dmmod,alpha)
            fmin = rb_norm(rbmin,rc,rho_s,r_set,msm,dmmod,alpha)
        else:
            fmax = rb_norm_nfw(rbmax,rc,rho_s,rv,msm)
            fmid = rb_norm_nfw(rbmid,rc,rho_s,rv,msm)
            fmin = rb_norm_nfw(rbmin,rc,rho_s,rv,msm)
        if(abs(fmax) <= 1e-5):
            found = True 
            rbf = rbmax
        elif(abs(fmin) <= 1e-5):
            found = True 
            rbf = rbmin
        elif(abs(fmid) <= 1e-5):
            found = True 
            rbf = rbmid
        n += 1
    return rbf

def find_alpha(rv,rc,rho_s,rhoc,M):
    found  = False
    tol = 1e-3
    dmmod = -1
    amax = 0.9
    amin = 1e-3
    amid = 0.5*(amax + amin)
    fmax = 1.0/astrophysics.rho_volume_int(rv,rc,1,dmmod,alpha=amax)*rho_s*rhoc/M*4.0*pi - 1.0
    fmid = 1.0/astrophysics.rho_volume_int(rv,rc,1,dmmod,amid)*rho_s*rhoc/M*4.0*pi - 1.0
    fmin = 1.0/astrophysics.rho_volume_int(rv,rc,1,dmmod,amin)*rho_s*rhoc/M*4.0*pi - 1.0
    f = amid
    n = 0

    while(n < 50 and (not found)):
        if(fmin > -tol and fmin < tol):
            found = True
            f = amin
        if(fmid > -tol and fmid < tol):
            found = True
            f = amid
        if(fmax > -tol and fmax < tol):
            found = True
            f = amax
        if(fmin > 0.0 and fmid > 0.0):
            amin = amid
            amid = 0.5*(amax + amin)
        if(fmax < 0.0 and fmid < 0.0):
            amax = amid
            amid = 0.5*(amax + amin)
        fmax = 1.0/astrophysics.rho_volume_int(rv,rc,1,dmmod,alpha=amax)*rho_s*rhoc/M*4.0*pi - 1.0
        fmid = 1.0/astrophysics.rho_volume_int(rv,rc,1,dmmod,amid)*rho_s*rhoc/M*4.0*pi - 1.0
        fmin = 1.0/astrophysics.rho_volume_int(rv,rc,1,dmmod,amin)*rho_s*rhoc/M*4.0*pi - 1.0
        n += 1
    return f
            
def rbar_nfw(cs):
    a = 1.0/3 + 0.5*cs + 0.2*cs**2
    b = 0.5 + 2.0/3*cs + 0.25*cs**2
    return cs*a/b

def rsig_nfw(cs):
    a = 0.25 + 0.4*cs + 1.0/6*cs**2
    b = 0.5 + 2.0/3*cs + 0.25*cs**2
    avgr2 =  cs**2*a/b
    avgr = rbar_nfw(cs)
    return sqrt(avgr2 - avgr**2)
