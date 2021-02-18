#cython: language_level=3
import numpy as np

def column(matrix,i):
    return np.array([row[i] for row in matrix])

class ps_env:
    def __init__(self,cosmo,zmax=10,mmin=1e13,mmax=1e17,mref=1e15,mn=201,zn=201,sigv=3e-26,numerical=0,counts=0,flux=False):
        self.cosmo = cosmo
        self.h = cosmo.h
        self.w_m = cosmo.w_m
        self.w_l = cosmo.w_l
        self.sigv = sigv
        self.zmax = zmax
        self.mmin = mmin
        self.mmax = mmax
        self.mn = mn
        self.zn = zn
        self.flux = flux
        self.mref = mref
        self.mset = np.logspace(np.log10(mmin),np.log10(mmax),num=mn)/mref
        self.zset = np.linspace(1e-5,zmax,num=zn)
        self.numerical = numerical
        self.counts = counts
        if(numerical == 1):
            self.grad_sig = dsigdr(self.mset*mref,self.cosmo)
        else:
            self.grad_sig = None
        self.Nnset = None
        self.Naset = None
        self.dfdm = None
        self.fset = []
        self.flux_array = None

    def get_numerical(self):
        nset = []
        for i in range(0,self.zn):
            lnset = np.zeros(self.mn)
            for j in range(0,self.mn):
                lnset[j] = N_rv(self.mset[j],self.zset[i],self.cosmo,2,self.grad_sig[j])
            nset.append(lnset)
        self.Nnset = np.array(nset)

    def get_analytical(self):
        aset = []
        for i in range(0,self.zn):
            lnset = np.zeros(self.mn)
            lnset = Nm_a(self.mset,1,self.zset[i],self.cosmo,2)
            aset.append(lnset)
        self.Naset = np.array(aset)

    def get_dfdm(self):
        self.dfdm = []
        for nu_index in range(0,len(self.flux_array[0][0])):
            dfdm = np.zeros((self.zn,self.mn),dtype=np.float)
            for i in range(0,self.zn):
                dfdm[i][:] = np.gradient(self.flux_array[i][:,nu_index],self.mset)[:]
            self.dfdm.append(dfdm)

    def read_flux(self,flux_file):
        try:
            ff = open(flux_file,"r")
        except IOError:
            print("No file: "+flux_file+" found!")
            sys.exit(2)
        fset = []
        zset = []
        mset = []
        zc = 0
        mc = 0
        loop_set = []
        c = 0
        for line in ff:
            s = line.rstrip().split(" ")
            try:
                if(zset == []):
                    zset.append(float(s[0]))
                    dl = dist_luminosity(float(s[0]),self.cosmo)
                elif(zset[zc] < float(s[0])):
                    zset.append(float(s[0]))
                    dl = dist_luminosity(float(s[0]),self.cosmo)
                    zc += 1    
                    fset.append(loop_set)
                    loop_set = []
                if(mset == []):
                    mset.append(float(s[1]))
                elif(mset[mc] < float(s[1])):
                    mset.append(float(s[1]))
                    mc += 1    
                floop = []
                for i in range(2,len(s)):
                    try:
                        if(self.flux):
                            flux_fac = 1.0
                        else:
                            flux_fac = 4*np.pi*dl**2
                        floop.append(float(s[i])*self.sigv*flux_fac)
                    except TypeError:
                        print(s[i])
                        sys.exit(2)
                loop_set.append(floop)
            except IndexError:
                print("Line: "+line+" of File: "+flux_file+" incorrectly formatted!")
                sys.exit(2)
        fset.append(loop_set)
        self.mset = np.array(mset)
        self.mn = len(mset)
        self.zset = np.array(zset)
        self.zn = len(zset)
        self.flux_array = np.array(fset)

    def get_flux_count(self,nu_index,fstar):
        self.get_dfdm()
        nm = np.zeros(self.zn)
        for i in range(0,self.zn):
            intp = sp.interp1d(self.mset,self.dfdm[nu_index][i])
            j = get_m_from_f(fstar,i,nu_index,self)
            if(j == -1):
                nm[i] = 0.0
            else:
                nm[i] = Nm_a(j/self.mref,1,self.zset[i],self.cosmo,2)*(intp(j))**(-1)*dvdz(self.zset[i],self.cosmo)
        return nm

    def get_count(self,M):
        weights = np.ones(self.mn)
        weights[self.mset < M] = 0
        nmi = np.zeros(self.zn)
        dz = np.zeros(self.zn)
        for i in range(0,self.zn):
            dz[i] = dvdz(self.zset[i],self.cosmo)
            nmi[i] = tools.Integrate(self.Naset[i]*weights,self.mset)
        return tools.Integrate(nmi*dz,self.zset)

    def get_z_count(self,M_index,z_index):
        return self.Naset[z_index][M_index]*dvdz(self.zset[z_index],self.cosmo)

    def get_zf_count(self,M_index,z_index,nu_index):
        return self.Naset[z_index][M_index]*dvdz(self.zset[z_index],self.cosmo)*self.flux_array[z_index][M_index][nu_index]

    def get_av_flux(self):
        nu_set = np.zeros(len(self.flux_array[0][0]))
        for i in range(0,len(self.flux_array[0][0])):
            jint = np.zeros(self.zn)
            for j in range(0,self.zn):
                int_set = self.flux_array[j][:,i]*Nm_a(self.mset/self.mref,1,self.zset[j],self.cosmo,2)
                jint[j] = tools.Integrate(int_set,self.mset)*dvdz(self.zset[j],self.cosmo)
            nu_set[i] = tools.Integrate(jint,self.zset)
        return nu_set

    def isotropic_intensity(self,max_flux):
        #print self.mn,len(self.flux_array[0]),self.zn,len(self.flux_array)
        nu_set = np.zeros(len(self.flux_array[0][0]))
        nu = np.logspace(1,5,num=len(self.flux_array[0][0]))*1e6
        for i in range(0,len(self.flux_array[0][0])):
            jint = np.zeros(self.zn)
            for j in range(0,self.zn):
                dl = dist_luminosity(self.zset[j],self.cosmo)
                if(self.flux):
                    flux_fac = 4.0*np.pi*dl**2
                else:
                    flux_fac = 1.0
                weight = np.zeros(self.mn)
                for l in range(0,self.mn):
                    if(self.flux_array[j][l][i]/(dl**2*4*np.pi) <= max_flux):
                        weight[l] = 1.0
                int_set = column(self.flux_array[j],i)*Nm_a(self.mset/self.mref,1,self.zset[j],self.cosmo,2)*weight
                jint[j] = 0.25/np.pi*tools.Integrate(int_set,self.mset)*dvdz(self.zset[j],self.cosmo)*(1+self.zset[j])/dl**2*flux_fac
            nu_set[i] = tools.Integrate(jint,self.zset)#*nu[i]
        return nu_set

    def get_flux_dist(self,z_index):
        outf = open("mf_z"+str(self.zset[z_index])+"_flux_dist.out","w")
        nmf = np.zeros(len(self.dfdm[z_index]))
        for i in range(0,len(self.dfdm[z_index])):
            nmf[i] = Nm_a(self.mset[i]/self.mref,1,self.zset[z_index],self.cosmo,2)*(self.dfdm[z_index][i])**(-1)
            #print Nm_a(self.dfdm[0][i],1,z,self.h,self.w_m,self.w_l,2), (self.dfdm[1][i]*3e-27)**(-1)
            outf.write(str(self.dfdm[2][i])+" "+str(nmf[i])+"\n")
        outf.close()
        return nmf

#order the arguments according to the flags they were given with

#diff is the diffusion flag, 0 -> no diffusion
#rf_flag is flux integrationf flag, 1 -> radius set by input files, 2 -> virial radius

def yd(h,w_m,w_b):
    #this calculates the expansion factor since matter-radiation equality
    Tcmb = 2.728
    theta = 2.728/2.7
    b1 = 0.313*(w_m*h**2)**(-0.419)*(1+0.607*(w_m*h**2)**(0.674)) #Eisenstein and Hu
    b2 = 0.238*(w_m*h**2)**(0.223)
    zd = 1291*(w_m*h**2)**(0.251)*(1+b1*(w_b*h**2)**b2)/(1 + 0.659*(w_m*h**2)**(0.828)) #compton drag z
    zeq = 2.5e4*w_m*h**2*theta**(-4) #matter-rad equality z
    return (1+zeq)/(1+zd)

def cms(h,w_m,w_b):
    #co-moving sound propagation distance prior to zd
    s = 44.5*np.log(9.83/(w_m*h**2))/np.sqrt(1+10*(w_b*h**2)**0.75) #Mpc
    return s

def nfw_fourier_a(k,M,z,cosmo):
    rv = rvir(M,z,cosmo)
    cv = cvir_sig(M,z,cosmo)
    rc = rv/cv
    u = (np.sin(k*rc)*(Si((1+cv)*k*rc)-Si(k*rc)) - np.sin(cv*k*rc)/((1+cv)*k*rc) + np.cos(k*rc)*(Ci((1+cv)*k*rc) - Ci(k*rc)))
    return u*4*np.pi*rc**3/M*rhos(cv,z,cosmo)*rho_crit(z,cosmo)

def nfw_fourier(k,M,z,cosmo,w_d,ex):
    rv = rvir(M,z,cosmo)
    cv = cvir_sig(M,z,cosmo)
    rc = rv/cv
    rset = np.logspace(np.log10(rc*1e-3),np.log10(rv),num=50)
    rho = rho_dm(rset,rc,1,0.18)*rhos(cv,z,cosmo)#*rho_crit(z,h,w_m,w_l)
    rint = 2*4*np.pi*rset**2*np.cos(2*np.pi*k*rset)*rho**ex
    rn = 4*np.pi*rset**2*rho**ex
    return tools.Integrate(rint,rset)/tools.Integrate(rn,rset)

def Si(x):
    xset = np.linspace(1e-3,x,num=50)
    yset = np.sin(xset)/xset
    return tools.Integrate(yset,xset)

def Ci(x):
    xset = np.logspace(np.log10(x),np.log10(x*1e4),num=50)
    yset = np.cos(xset)/xset
    return -tools.Integrate(yset,xset)

def lin_bias(M,z,cosmo):
    r = rvir_ps(M,0.0,cosmo)
    rmin = rvir_ps(1e6,0.0,cosmo)
    sigma = glinear(z,cosmo)**2*sigma_l_pl(r,0,1.0,rmin,z,cosmo)
    sigma8 = sigma_l_pl(8/cosmo.h,0,1.0,rmin,0.0,cosmo)/0.897**2
    nu = 1.686**2/sigma*sigma8
    q = 1
    p = 0.5 #q = 1, p = 0.5 is P&S 
    e1 = (q*nu-1)/1.686
    E1 = 2*p/1.686/(1+(q*nu)**p)
    return 1 + e1 + E1

def transfer(k,h,w_m,w_b):
    theta = 2.728/2.7
    s = cms(h,w_m,w_b)
    f_b = w_b/w_m
    f_d = (w_m - w_b)/w_m
    p_b = 0.25*(5 - np.sqrt(1+24*f_b))
    p_d = 0.25*(5 - np.sqrt(1+24*f_d))
    p_bd = 0.25*(5 - np.sqrt(1+24*f_d+24*f_b))
    av = f_d/(f_b+f_d)*(5-2*(p_d + p_bd))/(5-4*p_bd)
    av = av*(1-0.553*f_b+0.126*f_b**3)*(1+yd(h,w_m,w_b))**(p_bd-p_d)
    av = av*(1 + 0.5*(p_b-p_bd)*(1+1/(3-4*p_d)/(7-4*p_bd))*(1+yd(h,w_m,w_b))**(-1))
    zeq = 2.5e4*w_m*h**2*theta**(-4) #matter-rad equality z
    #q = k/19.0*(w_m*h**2*1e4)**(-0.5)*(1+zeq)**(-0.5)
    q = k/19.0*(w_m*h**2*1e4)**(-0.5)*(1+zeq)**(-0.5)
    Geff = w_m*h**2*(np.sqrt(av) + (1-np.sqrt(av))/(1+(0.43*k*s)**4))
    qeff = k*theta**2/Geff #Mpc^-1
    C = 14.4 + 325/(1+60.5*qeff**(1.08))
    beta = (1.0 - 0.949*f_b)**(-1)
    L = np.log(np.exp(1.0) + 1.84*beta*np.sqrt(av)*qeff)
    return L/(L + C*qeff**2)

def pspec_mf(k,n,z,cosmo,w_b):
    D1z = glinear(z,cosmo)
    D10 = glinear(0,cosmo)
    #dh = 4.2e-5
    nb = 1 - n
    dh = 1.94e-5*cosmo.w_m**(-0.785-0.05*np.log(cosmo.w_m))*np.exp(-0.95*nb-0.169*nb**2)
    c = 2.99792458e5 #km s^-1
    return dh**2*(c/(cosmo.h*100))**(3+n)*k**n*transfer(k,cosmo.h,cosmo.w_m,w_b)**2*2*np.pi**2*D1z**2/D10**2

def p_1h_2h(k,mset,z,cosmo,w_b):
    #this returns the 1-halo and 2-halo contributions to the power spectrum
    #the return data is 2-element array with 1h -> 0 and 2h -> 1
    pset = np.zeros(len(mset))
    bset = np.zeros(len(mset))
    nmset = Nm_a(mset/1e15,1,z,cosmo,2)
    for i in range(0,len(mset)):
        #we call nfw fourier with a 2 for rho^2 - the WIMP pair density
        pset[i] = np.abs(nfw_fourier(k,mset[i],z,cosmo,cosmo.w_m-w_b,2))
        bset[i] = lin_bias(mset[i],z,cosmo)#*(mset[i]/rho_crit(z,h,w_m,w_l)/omega_m(z,h,w_m,w_l))
    return np.array([pset**2*nmset,pset*bset*nmset])

def angular_spectrum(ps,l,n,Lav,nu_index,m_x,max_flux,w_d):
    k = np.zeros(ps.zn)
    dz = np.zeros(ps.zn)
    dl = np.zeros(ps.zn)
    for i in range(0,ps.zn):
        k[i] = l/dist_co_move(ps.zset[i],ps.cosmo) #co-moving wavenumber
        dz[i] = dvdz_cl(ps.zset[i],ps.cosmo,ps.flux) #redshift volume
        dl[i] = dist_luminosity(ps.zset[i],ps.cosmo)
    rhoc = rho_crit(ps.zset,ps.cosmo)
    nu = np.logspace(1,5,num=10)
    w_b = ps.w_m-w_d
    c = 2.995e10 #c in cm
    kb = 1.3806488e-23*1e7 #kb in erg K^-1
    clm = np.zeros(ps.zn)
    rmin = rvir(ps.mset[0],0.0,ps.cosmo)
    sigma8 = checkSigma8(ps.cosmo)/0.897**2 #ensure that psz is normalised to Sigma8
    jint = np.zeros(ps.zn)
    for j in range(0,ps.zn):
        psnl = p_1h_2h(k[j],ps.mset,ps.zset[j],ps.cosmo,w_b) #1 and 2 halo contributions in 2-e array
        psz = pspec_mf(k[j],n,ps.zset[j],ps.cosmo,w_b)#/sigma8 #linear power spectrum
        weight = np.zeros(ps.mn)
        for i in range(0,ps.mn):
            if(ps.flux_array[j][i][nu_index]/(4*np.pi*dl[j]**2) <= max_flux):
                weight[i] = 1.0
        dndE = ps.flux_array[j][:,nu_index]*weight #luminosity-mass spectrum for given frequency and z
        int_set = np.array([dndE**2*psnl[0],dndE*psnl[1]])
        jint[j] = tools.Integrate(int_set[0],ps.mset) + tools.Integrate(int_set[1],ps.mset)**2*psz #m integration
        #print dz[j],tools.Integrate(int_set[0],ps.mset), tools.Integrate(int_set[1],ps.mset)**2*psz
    #return tools.Integrate(jint*dz,ps.zset)/(4.0*np.pi)**2*c**4/kb**2*0.25/(nu[nu_index]*1e6)**4*(1e-23)**2
    return tools.Integrate(jint*dz,ps.zset)/(4.0*np.pi)**2/Lav[nu_index]**2

def checkSigma8(cosmo):
    #the average excess mass with a sphere of radius r
    #z is redshift, h is H(0) in 100 Mpc s^-1 km^-1, w_m is matter density parameter at z = 0
    #r is sphere radius (Mpc), rc is the cutoff radius, rmin is the minimum radius
    n = 1001
    pn = 1
    r = 8/cosmo.h
    kmax = 1.0/rvir(1e-6,0,cosmo)
    kmin = 1.0e-12*kmax
    pset = np.zeros(n,dtype=np.float)
    kset = np.logspace(np.log10(kmin),np.log10(kmax),num=n)
    for i in range(0,n):
        pset[i] = pspec_mf(kset[i],pn,0.0,cosmo,cosmo.w_m-0.048)
    kint = 0.5*kset**(2)*window(kset*r)**2/np.pi**2*pset
    return tools.Integrate(kint,kset)
        

def Nm_a(M,n,z,cosmo,j):
    rhobar = rho_crit(0.0,cosmo)*omega_m(0.0,cosmo)/1e15
    #N0 = 1.8e-4*h**3
    N0 = 4.7e-4*cosmo.h**3 #using Mstar = M_8
    y = 2-(n+3)/6.0
    r8 = 8/cosmo.h
    b = 1.0
    dc = 1.0#1.686
    Mstar = 4.0/3.0*np.pi*(r8)**3*rhobar
    dz = glinear(z,cosmo)
    return N0*j/sqrt(2*np.pi)*(n+3)/6.0*dc*b/Mstar*(M/Mstar)**(-y)/dz*np.exp(-0.5*dc**2*b**2/dz**2*(M/Mstar)**((n+3)/3.0))

def get_m_from_f(fstar,z_index,nu_index,ps):
    intp = sp.interp1d(ps.flux_array[z_index][:,nu_index],ps.mset)
    try:    
        m = intp(fstar)
    except ValueError:
        m = -1
    return m

def dvdz(z,cosmo):
    dl = dist_luminosity(z,cosmo)
    w_k = 1 - cosmo.w_m - cosmo.w_l
    return 2.998e5/(cosmo.h*100)*dl**2/np.sqrt((1+z)**3*cosmo.w_m + w_k*(1+z)**2 + cosmo.w_l)/(1+z)**2

def dvdz_cl(z,cosmo,flux):
    dl = dist_luminosity(z,cosmo)
    w_k = 1 - cosmo.w_m - cosmo.w_l
    if(flux):  #if its a flux we correct it to a luminosity
        flux_fac = 4*np.pi*dl**2
    else:
        flux_fac = 1.0
    return 2.998e5/(cosmo.h*100)*dl**(-2)/np.sqrt((1+z)**3*cosmo.w_m + w_k*(1+z)**2 + cosmo.w_l)/(1+z)**2*flux_fac**2

def N_rv(M,z,cosmo,j,dsigr):
    M = M*1e15
    r = rvir_ps(M,0.0,cosmo)
    rhobar = rho_crit(z,cosmo)*omega_m(z,cosmo)
    rhobar0 = rho_crit(0.0,cosmo)*omega_m(0.0,cosmo)
    dz = glinear(z,cosmo)
    r8 = 8/h
    rcut = (3*1.0e-6/(4*np.pi*rhobar0))**(1.0/3)
    sig8 = sigma_l_pl(r8,0,rcut,0.01*rcut,0.0,cosmo)
    #b = 0.897**2/sig8
    b = 1.0/sig8
    sig = dz*sqrt(sigma_l_pl(r,0,rcut,0.01*rcut,0.0,cosmo)*b)
    dc = 1.0#1.686
    dnu = dnudm(r,z,cosmo,dc,sig,b,dsigr)
    nuc = dc/sig
    return j/M*dnu*rhobar0*np.exp(-nuc**2*0.5)/sqrt(2*np.pi)*1e15
    

def dnudm(r,z,cosmo,dc,sig,b,dsigr):
    rhobar = rho_crit(z,cosmo)*omega_m(z,cosmo)
    rhobar0 = rho_crit(0.0,cosmo)*omega_m(0.0,cosmo)
    rcut = (3*1.0e-6/(4*np.pi*rhobar0))**(1.0/3)
    dz = glinear(z,cosmo)
    dsig = -dsigr*sqrt(b)/sig**2*dz
    Mvir = 4*np.pi*rhobar0*r**3/3.0 
    drdm = 0.206783/(Mvir/rhobar0)**(2.0/3)/rhobar0
    return dc*drdm*dsig

def dsigdr(M,cosmo):
    dr = np.zeros(len(M))
    sigr = np.zeros(len(M))
    sigrm1 = np.zeros(len(M))
    sigrm2 = np.zeros(len(M))
    sigrp1 = np.zeros(len(M))
    sigrp2 = np.zeros(len(M))
    rcut = (3*1.0e-6/(4*np.pi*omega_m(0.0,cosmo)*rho_crit(0.0,cosmo)))**(1.0/3)
    rset = rvir_ps(M,0.0,cosmo)
    for j in range(0,len(rset)):
        dr[j] = (rset[len(rset)-1]-rset[0])/100.0
        sigr[j] = sqrt(sigma_l_pl(rset[j],0,rcut,0.01*rcut,0.0,cosmo))
        sigrm1[j] = sqrt(sigma_l_pl(rset[j]-dr[j],0,rcut,0.01*rcut,0.0,cosmo))
        sigrp1[j] = sqrt(sigma_l_pl(rset[j]+dr[j],0,rcut,0.01*rcut,0.0,cosmo))
        sigrm2[j] = sqrt(sigma_l_pl(rset[j]-2*dr[j],0,rcut,0.01*rcut,0.0,cosmo))
        sigrp2[j] = sqrt(sigma_l_pl(rset[j]+2*dr[j],0,rcut,0.01*rcut,0.0,cosmo))
    dsig = finite_diff(sigrm2,sigrm1,sigr,sigrp1,sigrp2,dr)
    return dsig

def dw2dr(k,r):
    x = k*r
    w = window(x)
    #return 6*w*k/x**2*np.sin(x) - 18*w**2/r
    return -18.0*k*(x*np.cos(x) - np.sin(x))*(3*x*np.cos(x) + (x**2-3)*np.sin(x))/x**7

def finite_diff(sm2,sm1,s,sp1,sp2,dx):
    return (sm2 - sm1*8 + 8*sp1 - sp2)/12.0/dx

def read_mf(in_file):
    #this function reads in the input variables for the cosmology
    ps = None
    try:
        inf = open(in_file,"r")
        count = 1
        d = []
        for line in inf:
            if(count%2 == 0):
                s = line.rstrip().split(" ")
                for x in s:
                    d.append(x)
            count += 1
        cosmo = environments.cosmology_env(h=float(d[0]),w_m=float(d[1]),w_l=float(d[2]))
        ps = ps_env(cosmo,zmax=float(d[3]),mmin=float(d[4]),mmax=float(d[5]),mref=float(d[6]),mn=int(d[7]),zn=int(d[8]),numerical=int(d[9]),counts=int(d[10]))
    except IOError:
        tools.fatal_error("Invalid input file string: "+in_file)
    return ps
