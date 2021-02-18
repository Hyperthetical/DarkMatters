#cython: language_level=3
from numpy import *
import sys,platform
from scipy.integrate import simps as integrate
from emm_tools.tools_emm import progress
#from numba import jit
from subprocess import call
from emm_tools.electrons_crank import getElectrons_numeric
#@jit
#def integrate(x,y):
#    lx = log10(x)
#    ly = log10(y)
#    h = (lx[-1]-lx[0])/len(x)
#    w = 1.0/3*ones(len(x))
#    indices = arange(0,len(x))
#    w = where(indices%2==0,2.0/3,w)
#    w = where(indices%2!=0,4.0/3,w)
#    w[-1] = 1.0/3
#    w[0] = 1.0/3
#    return sum(h*w*10**(ly+lx))

#read the output from the c routine
def read_electrons_c(infile,halo,phys,sim):
    """
    Read the output from the c executable that finds equilibrium electon distributions
        ---------------------------
        Parameters
        ---------------------------
        infile   - Required : file path to c output (String)
        halo     - Required : halo envionment (halo_env)
        phys     - Required : physical environment (phys_env)
        sim      - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        2D array of floats, electron equilibrium distributions (phys.e_bins x sim.n)
    """
    try:
        inf = open(infile,"r")
    except:
        return None
    line = inf.readline().strip().split()
    eArray = array(line,dtype=float)
    n = sim.n
    k = len(phys.spectrum[0])
    electrons = zeros((k,n),dtype=float)
    for i in range(0,k):
        for j in range(0,n):
            electrons[i][j] = eArray[i*n + j]
    return electrons

#write input file for c executable
def write_electrons_c(outfile,halo,phys,sim):
    """
    Write the input file for the c executable that finds equilibrium electon distributions
        ---------------------------
        Parameters
        ---------------------------
        outfile  - Required : file inputs to c executable go to (String)
        halo     - Required : halo envionment (halo_env)
        phys     - Required : physical environment (phys_env)
        sim      - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        None
    """
    outf = open(outfile,"w")
    outf.write(str(len(phys.spectrum[0]))+" "+str(sim.n)+" "+str(sim.ngr)+"\n")
    for r in halo.r_sample[0]:
        outf.write(str(r)+" ")
    outf.write("\n")
    for r in halo.r_sample[1]:
        outf.write(str(r)+" ")
    outf.write("\n")
    for x in phys.spectrum[0]:
        outf.write(str(x)+" ")
    outf.write("\n")
    for x in phys.spectrum[1]:
        outf.write(str(x)+" ")
    outf.write("\n")
    for r in halo.rho_dm_sample[0]:
        outf.write(str(r)+" ")
    outf.write("\n")
    for r in halo.rho_dm_sample[1]:
        outf.write(str(r)+" ")
    outf.write("\n")
    for x in halo.b_sample:
        outf.write(str(x)+" ")
    outf.write("\n")
    for x in halo.ne_sample:
        outf.write(str(x)+" ")
    outf.write("\n")
    outf.write(str(halo.z)+" "+str(phys.mx)+" "+str(phys.lc)+" "+str(phys.delta)+" "+str(halo.bav)+" "+str(halo.neav)+"\n")
    outf.write(str(phys.diff)+" "+str(int(phys.ISRF))+" "+str(phys.d0)+" "+str(halo.mode_exp))
    outf.close()

#run the c executable with a written infile and retrieve output
def electrons_from_c(outfile,infile,halo,phys,sim):
    """
    Prepare the input file, run the c executable that finds equilibrium electon distributions and retrieve the output
        ---------------------------
        Parameters
        ---------------------------
        outfile  - Required : file inputs to c executable go to (String)
        infile   - Required : file path to c output (String)
        halo     - Required : halo envionment (halo_env)
        phys     - Required : physical environment (phys_env)
        sim      - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        2D array of floats, electron equilibrium distributions (phys.e_bins x sim.n)
    """
    write_electrons_c(outfile,halo,phys,sim)
    try:
        if platform.system() == "Linux":
            call([sim.exec_electron_c+" "+outfile+" "+infile],shell=True)#,cwd=cdir)
        else:
            call([sim.exec_electron_c,outfile,infile],shell=True)#,cwd=cdir)
    except FileNotFoundError:
        return None
    electronData = read_electrons_c(infile,halo,phys,sim)
    return electronData

#@jit
def eloss_vector(E_vec,B,ne,z,ISRF=0):
    """
    Calculates a vectorised form of the energy loss function b(E)
        ---------------------------
        Parameters
        ---------------------------
        E_vec  - Required : set of energy values for b(E) (float)
        B      - Required : average magnetic field strength (float)
        ne     - Required : average gas density (float)
        z      - Required : redshift of the halo bing considered (float)
        ISRF   - Optional : flag whether to include ISRF in inverse-compton losses (int, 1 or 0)
        ---------------------------
        Output
        ---------------------------
        1D float array (sim.n)
    """
    n = ne#*(1+z)**3
    me = 0.511e-3 #GeV -> E_vec comes in as E/me
    coeffs = array([6.08e-16+0.25e-16*(1+z)**4,0.0254e-16,6.13e-16,4.7e-16],dtype=float)
    if ISRF == 0:
        coeffs[0] = 0.25e-16*(1+z)**4 #only CMB used so it scales with z
    if ne == 0.0: 
        eloss_tot = coeffs[0]*(me*E_vec)**2 + coeffs[1]*(me*E_vec)**2*B**2 + coeffs[2]*n*(1+log(E_vec/n)/75.0)+ coeffs[3]*n*E_vec*me
    else:
        eloss_tot = coeffs[0]*(me*E_vec)**2 + coeffs[1]*(me*E_vec)**2*B**2 
    return eloss_tot/me #make it gamma s^-1 units

#@jit
def eloss_loop(E,B,ne,z):
    #vectorised energy loss calculation
    #calculates b(E) for every value in E_set
    #E_vec is the set of energies, B is the mean magnetic field, ne is the mean plasma density, z is redshift
    n = ne#*(1+z)**3
    return 1.37e-20*E**2*(1+z)**4 + 1.30e-21*E**2*B**2 + 6.13e-16*n*(1+log(E/n)/75.0)+ 1.51e-16*n*(0.36+log(E/n))

#@jit
def green_integrand_vector(rpr,rn,dv,rhosq):
    """
    Calculates and returns the integrand of the Green's function for electron diffusion
        ---------------------------
        Parameters
        ---------------------------
        rpr   - Required : set of r points in the integrand of the Green's function (float)
        rn    - Required : bounday condition poisiton of image charge (float)
        dv    - Required : diffusion scale parameter (float)
        rhosq - Required : ratio of rho^2 at r and each rpr point (float)
        ---------------------------
        Output
        ---------------------------
        1D array of floats (len(rpr)) 
    """
    return rpr/rn*(exp(-(rpr-rn)**2/(4.0*dv))-exp(-(rpr+rn)**2/(4.0*dv)))*rhosq

#@jit
def Green_vector(r_set,r,rhosq,dv,diff):
    """
    Calculates and returns a vector of Green's function values
        ---------------------------
        Parameters
        ---------------------------
        r_set - Required : set of all r points in Green integrand (float)
        r     - Required : position to calculate at (float)
        dv    - Required : diffusion scale parameter (float)
        rhosq - Required : ratio of rho^2 at r and each r_set point (float array)
        diff  - Required : diffusion flag (0 or 1)
        ---------------------------
        Output
        ---------------------------
        1D float array (phys.e_bins)
    """
    m = len(dv)
    G = zeros(m)
    if(diff == 0):
        G[dv != -1] = 1.0
    else:
        n = len(r_set)
        rh = r_set.max()
        images = 33 #image charges for green function solution -> should be odd
        image_set = arange(-(images-1)/2,(images-1)/2,dtype=int)

        for i in range(0,m):
            if(dv[i] == 0.0):
                G[i] = 1.0 #this is a no diffusion case or E = Epr case
            elif(dv[i] == -1): #this exists to limit the domain of the Green's function to between relevant energies
                G[i] = 0.0
            else:
                k1 = 1.0/sqrt(4*pi*dv[i]) #coefficient
                for j in range(0,len(image_set)):
                    p = image_set[j]
                    rn = (-1.0)**p*r + 2.0*p*rh  #position of the image charge
                    G[i] = G[i] + k1*Green_p(r,rn,r_set,rhosq,dv[i],p)
    return G

#@jit
def Green_p(r,rn,r_set,rhosq,dv,p):
    """
    Calculates Green function contribution from an image charge p
        ---------------------------
        Parameters
        ---------------------------
        r     - Required : position to calculate at (float)
        r_set - Required : set of r points in the integrand of the Green's function (float)
        rn    - Required : bounday condition poisiton of image charge (float)
        dv    - Required : diffusion scale parameter (float)
        rhosq - Required : ratio of rho^2 at r and each rpr point (float)
        p     - Required : image charge p
        ---------------------------
        Output
        ---------------------------
        float
    """
    r_int = green_integrand_vector(r_set,rn,dv,rhosq)
    return (-1.0)**p*integrate(r_int,r_set)

def diffusion_constant(halo,phys):
    """
    This sets the phys.d0 value for the diffusion constant
        ---------------------------
        Parameters
        ---------------------------
        halo - Required : DM halo environment (halo_env)
        phys - Required : Physical environment (phys_env)
        ---------------------------
        Output
        ---------------------------
        None - assigned to phys.d0 [cm^2 s^-1]
    """
    d0 = (1.0/3.0)*(3e8)**(phys.delta-1.0)*(1.6e-19)**(phys.delta-2.0)*(1.6e-10)**(2.0-phys.delta)
    d0 = d0*(halo.bav*1e-10)**(phys.delta-2.0)*(phys.lc*3.09e19)**(phys.delta-1.0)*1e4
    phys.d0 = d0 

def equilibrium_p(halo,phys):
    """
    This is a wrapper function to fit into the API of my python code
        ---------------------------
        Parameters
        ---------------------------
        halo - Required : DM halo environment (halo_env)
        phys - Required : Physical environment (phys_env)
        ---------------------------
        Output
        ---------------------------
        2D float array (phys.e_bins x sim.n) 
    """
    return equilibrium_electrons(phys.spectrum[0],phys.spectrum[1],halo.r_sample,halo.rho_dm_sample,phys.mx,halo.mode_exp,halo.bav,halo.neav,halo.z,phys.lc,phys.delta,phys.diff,phys.d0,phys.ISRF)
    #return getElectrons_numeric(halo,phys)

#@jit#(nopython=True,parallel=True)
def equilibrium_electrons(E_set,Q_set,r_sample,rho_dm_sample,mx,mode_exp,b_av,ne_av,z,lc,delta,diff,d0,ISRF):
    """
    Calculates equilibrium electron distribution 2D array (energy)(position)
        ---------------------------
        Parameters
        ---------------------------
        E_set            - Required : an array of E/me, Lorentz gammas
        Q_set            - Required : electron generation function from chi-chi annihilation at each E_set value
        r_sample[0]      - Required : set of radial sampling values 
        r_sample[1]      - Required : set of radial sampling values for the Green's functions
        rho_dm_sample[0] - Required : WIMP pair density at each radial sampling value
        rho_dm_sample[1] - Required : WIMP pair density at each radial sampling value for the Green's functions sampling
        mx               - Required : WIMP mass in GeV
        mode_exp         - Required : 2.0 for annihilation, 1.0 for decay
        b_av             - Required : average magnetic field strength in uG
        ne_av            - Required : average plasma density in cm^-3
        z                - Required : redshift
        lc               - Required : turbulent length scale for B in kpc
        delta            - Required : power-law slope for the B field turbulence spectrum 5/3 Kolmogorov, 2 is Bohm
        diff             - Required : flag, 0 -> no diffusion, 1 -> diffusion
        ISRF             - Required : flag, 0 -> CMB IC only in energy-loss, 1 -> ISRF and CMB
        ---------------------------
        Output
        ---------------------------
        2D float array (phys.e_bins x sim.n)
    """
    #E_set = phys.spectrum[0]
    #Q_set = phys.spectrum[1]
    k = len(E_set) #number of energy bins
    n = len(r_sample[0])
    ngr = len(r_sample[1])

    loss = zeros(k)#,dtype=float)  #energy loss (E,r)
    electrons = zeros((k,n))#,dtype=float)  #energy loss (E,r)
    int_E = zeros(k)#,dtype=float)  
    int_E2 = zeros(k)#,dtype=float)  
    rhosq = zeros((n,ngr))#,dtype=float)
    me = 0.511e-3 #GeV

    Q_set[Q_set<0.0] = 0.0
    #msun converted to kg, convert to GeV, convert kpc to cm 
    nwimp0 = sqrt(1.458e-33)**mode_exp/mode_exp*(1.0/mx)**mode_exp  #non-thermal wimp density (cm^-3) (central)
    rhodm = nwimp0*rho_dm_sample[0]
    rhodm_gr = nwimp0*rho_dm_sample[1]
    #for i in range(0,k):
    #    loss[i] = eloss_loop(E_set[i],b_av,ne_av,z)
    loss = eloss_vector(E_set,b_av,ne_av,z,ISRF)
    for i in range(0,n):
        rhosq[i,] = rhodm_gr/rhodm[i]
        #for j in range(0,ngr):
        #    rhosq[i][j] = rhodm_gr[j]/rhodm[i] 
    #turbulent diffusion factors 
    #d0 = (1.0/3.0)*(3e8)**(delta-1.0)*(1.6e-19)**(delta-2.0)*(1.6e-10)**(2.0-delta)
    #d0 = d0*(b_av*1e-10)**(delta-2.0)*(lc*3.09e19)**(delta-1.0)*1e4   #cm^2 s^-1
    #print(d0,lc,b_av)
    d0 = exp(log(d0)-2.0*log(3.09e24))   #Mpc^2 s^-1 from cm^2 s^-1
    #d0 = 3.1e28*lc**(2.0/3)*bmu**(-1.0/3)
    if(diff == 1):
        vtab = make_vtab(E_set,d0,delta,loss,me)
    else:
        vtab = zeros(k)
    dv = zeros(k)
    for i in range(0,k): #loop over energies
        E = E_set[i]
        if(diff == 1):
            #print(diff)
            dv = -vtab + vtab[i]
            #print(dv)
        dv[E_set < E] = -1
        for j in range(0,n):   #loop of r
            r = r_sample[0][j]
            int_E = Q_set*Green_vector(r_sample[1],r,rhosq[j],dv,diff)  #diffusion integrand vectroised over E
            electrons[i][j] = 2.0*integrate(int_E,E_set)/loss[i]*rhodm[j] #the 2 is for electrons and positrons
            progress(i*n + j + 1,k*n)
    #sys.stdout.write("\n")
    return electrons              

def D(E_set,B,lc,delta):
    """
    Returns the diffusion function
        ---------------------------
        Parameters
        ---------------------------
        E_set - Required :energy domain
        B     - Required : mean magnetic field
        lc    - Required : minimum homogenuity scale for the field
        delta - Required : turbulence spectral index
        ---------------------------
        Output
        ---------------------------
        1D float array (phys.e_bins)
    """
    me = 0.511e-3
    d0= (1.0/3.0)*(3e8)**(delta-1.0)*(1.6e-19)**(delta-2.0)*(1.6e-10)**(2.0-delta)
    d0 = d0*(B*1e-10)**(delta-2.0)*(lc*3.09e19)**(delta-1.0)*1e4   #cm^2 s^-1
    d0 = exp(log(d0)-2.0*log(3.09e24))   #Mpc^2 s^-1
    dset = d0*(E_set*me)**(2.0-delta)
    return dset

#@jit
def get_v(E,E_set,d0,delta,loss,me): 
    """
    Returns the integrated energy variable for use in the Green's function
        ---------------------------
        Parameters
        ---------------------------
        E     - Required : minimal energy for integration domain, maximal is the wimp mass
        E_set - Required : energy domain
        d0    - Required : diffusion constant in Mpc^2 s^-1
        delta - Required : turbulence spectral index
        loss  - Required : energy loss function at E in GeV s^-1
        me    - Required : electron mass in GeV 
        ---------------------------
        Output
        ---------------------------
        float
    """
    int_v = d0*(E_set*me)**(2.0-delta)/loss
    int_v[E_set<E] = 0.0 
    v = integrate(int_v,E_set)
    return v

#@jit
def make_vtab(E_set,d0,delta,loss,me):
    """
    Returns a table of integrated energy variables for use in the Green's function
        ---------------------------
        Parameters
        ---------------------------
        E_set - Required : energy domain
        d0    - Required : diffusion constant in Mpc^2 s^-1
        delta - Required : turbulence spectral index
        loss  - Required : energy loss function at E in GeV s^-1
        me    - Required : electron mass in GeV 
        ---------------------------
        Output
        ---------------------------
        1D float array (phys.e_bins)
    """
    k = len(E_set)
    vtab = zeros(k)#,dtype=float)
    for i in range(0,k):
        vtab[i] = get_v(E_set[i],E_set,d0,delta,loss,me)
    return vtab

#def print_loss_scales(M,ne,z):
    #this prints out time scales for diffusion and energy loss
    #for halo mass M, thermal plasma density ne and at redshift z
    #it uses the D(E) and energy_loss_vector functions
#    spectrum = tools.read_spectrum("/home/geoff/Coding/Python/wimp/it_data/pos_bb_60GeV.dat")
#    outfile = "/home/geoff/Coding/Python/wimp/bb60_m"+str(int(log10(M)))+"_b5_z"+str(z)+"_loss.data"
#    E_set = spectrum[0]
#    Q_set = spectrum[1]
#    loss = eloss_vector(E_set,5.0,ne,z)
#    delta = 5.0/3.0
#    h = 0.673
#    w_m = 0.315
#    rh = cosmology.rvir(M,z,h,w_m) 
#    d0 = D(E_set,5.0,1.0,delta)
#    out = open(outfile,"w")
#    for i in range(0,len(E_set)):
#        out.write(str(E_set[i])+" "+str(rh**2/d0[i])+" "+str(E_set[i]/loss[i])+"\n")
