#cython: language_level=3
from scipy.integrate import simps
from scipy.interpolate import interp1d
from numpy import *
import sys

def Integrate(y,x):
    #just how to call the method without importing scipy everywhere
    return simps(y,x)

def int_bessel(t):
    """
    Bessel integral approximation
        ---------------------------
        Parameters
        ---------------------------
        t - Required : float, array-like []
        ---------------------------
        Output
        ---------------------------
        1D float array-like []
    """
    return 1.25*t**(1.0/3.0)*exp(-t)*(648.0+t**2)**(1.0/12.0)

def weightedLineAvg(y,r,w=None,rmax=-1):
    if rmax == -1:
        rmax = r[-1]
    if w is None:
        w = ones(len(r),dtype=float)
    yintp = interp1d(r,y)
    wintp = interp1d(r,w)
    x = logspace(log10(r[0]),log10(rmax),num=100)
    ny = yintp(x)
    nw = wintp(x)
    return Integrate(ny*nw,x)/Integrate(nw,x)

def weightedVolAvg(y,r,w=None,rmax=-1):
    if rmax == -1:
        rmax = r[-1]
    if w is None:
        w = ones(len(r),dtype=float)
    yintp = interp1d(r,y)
    wintp = interp1d(r,w)
    x = logspace(log10(r[0]),log10(rmax),num=100)
    ny = yintp(x)
    nw = wintp(x)
    return Integrate(ny*nw*x**2,x)/Integrate(nw*x**2,x)

def rho_nfw_smooth(rho,r_set,rc,rsm):
    #returns NFW convolved with Gaussian smoothing on radius rsm
    #r_set -> position sampling
    #rc -> scale radius
    #rsm -> smoothing radius
    n = len(r_set)
    rho_int = zeros(n,dtype=float) #integration storage
    for i in range(0,n):
        r = r_set[i]
        for j in range(0,n):
            rpr = r_set[j]
            #convolve rhopr with a gaussian
            rho_int[j] = rhopr[j]*rpr*(exp(-0.5*(rpr-r)**2/rsm**2) - exp(-0.5*(rpr+r)**2/rsm**2))
        #integrate to smear out 
        rho[i] = simps(rho_int,r_set)/(sqrt(2.0*pi)*r*rsm)
    return rho


def fatal_error(err_string):
    """
    Display error string and exit program
        ---------------------------
        Parameters
        ---------------------------
        err_sting - Required : error message (String)
        ---------------------------
        Output
        ---------------------------
        None
    """
    print("################################################")
    print("                   Fatal Error")
    print("################################################")
    print(err_string)
    sys.exit(2)
    
def help_menu():
    """
    Display help menu - not implemented!
    """
    print("================================================")
    print(" WIMP Calculator Help Menu")
    print("================================================")

def write(log,write_mode,sim,phys,cosm,halo):
    #writes a calculation summary to a flagged output stream
    #log is a flag to send the data to a file if it contains string or to screen if None
    #remaining values come from wimp_calculation.py
    """
    Write calculation data to a target output
        ---------------------------
        Parameters
        ---------------------------
        log        - Required : log file name (if None uses stdout) (String or None)
        write_mode - Required : 'flux' displays all information (String)
        sim        - Required : simulation environment (simulation_env)
        phys       - Required : physical environment (physical_env)
        cosm       - Required : cosmology environment (cosmology_env)
        ---------------------------
        Output
        ---------------------------
        Writes to a file or stdout
    """
    if(log is None):
        outstream = sys.stdout
    else:
        outstream = open(log,"w")
    end = "\n"
    if log is None:
        prefix = ""
    else:
        prefix = "#"
    outstream.write(prefix+'======================================================'+end)
    outstream.write(prefix+'Run Parameters'+end)
    outstream.write(prefix+'======================================================'+end)
    #outstream.write(prefix+'Flux Calculation Flag: '+str(sim.flux_flag)+end)
    #outstream.write(prefix+'Surface Brightness Calculation Flag: '+str(sim.sb_flag)+end)
    #outstream.write(prefix+'Mass Loop Calculation Flag: '+str(sim.loop_flag)+end)
    outstream.write(prefix+"Output directory: "+sim.out_dir+end)
    if halo.name != None:
        outstream.write(prefix+'Halo Name: '+str(halo.name)+end)
    elif(write_mode == "flux" and halo.ucmh == 0 and (not halo.mvir is None)):
        outstream.write(prefix+'Halo Mass Code: m'+str(int(log10(halo.mvir)))+end)
    #outstream.write(prefix+'Field File Code: b'+str(int(phys.b0))+"q"+str(phys.qb)+end)
    outstream.write(prefix+"Frequency Samples: "+str(sim.num)+end)
    outstream.write(prefix+"Minimum Frequency Sampled: "+str(sim.flim[0])+" MHz"+end)
    outstream.write(prefix+"Maximum Frequency Sampled: "+str(sim.flim[1])+" MHz"+end)
    outstream.write(prefix+"Radial Grid Intervals: "+str(sim.n)+end)
    outstream.write(prefix+"Green's Function Grid Intervals: "+str(sim.ngr)+end)
    outstream.write(prefix+'Minimum Sampled Radius: '+str(halo.r_sample[0][0])+' Mpc'+end)
    outstream.write(prefix+'Maximum Sampled Radius: '+str(halo.r_sample[0][sim.n-1])+' Mpc'+end)
    outstream.write(prefix+'======================================================'+end)
    outstream.write(prefix+'Dark Matter Parameters: '+end)
    outstream.write(prefix+'======================================================'+end)
    outstream.write(prefix+'WIMP mass: '+str(phys.mx)+' GeV'+end)
    if halo.mode_exp == 2.0 and not halo.mode == "special":
        outstream.write(prefix+'Annihilation channels used: '+str(phys.channel)+end)
    else:
        outstream.write(prefix+'Decay channels used: '+str(phys.channel)+end)
    if not halo.mode == "special":
        outstream.write(prefix+'Branching ratios used: '+str(phys.branching)+end)
    outstream.write(prefix+"Particle physics model label: "+str(phys.particle_model)+end)
    outstream.write(prefix+'======================================================'+end)
    outstream.write(prefix+'Halo Parameters: '+end)
    outstream.write(prefix+'======================================================'+end)
    outstream.write(prefix+'Redshift z: '+str(halo.z)+end)
    if halo.ucmh == 0:
        if(halo.dm == 1):
            outstream.write(prefix+'Halo Model: NFW'+end)
        elif(halo.dm == 2):
            outstream.write(prefix+'Halo Model: Cored (Burkert)'+end)
        elif(halo.dm == -1):
            outstream.write(prefix+"Halo Model: Einasto, Alpha: "+str(halo.alpha)+end)
        elif(halo.dm == 3):
            outstream.write(prefix+'Halo Model: Cored (Isothermal)'+end)
    else:
        outstream.write(prefix+'Halo Model: Ultra-compact'+end)
        #outstream.write(prefix+'Phase Transition: '+halo.phase+end)
    if(write_mode == "flux"):
        outstream.write(prefix+'Virial Mass: '+str(halo.mvir)+" Solar Masses"+end)
        if not halo.Dfactor is None:
            outstream.write(prefix+'Dfactor: '+str(halo.Dfactor)+" GeV cm^-2"+end)
        if not halo.J is None:
            outstream.write(prefix+'Jfactor: '+str(halo.J)+" GeV^2 cm^-5"+end)
        if halo.ucmh == 0:
            outstream.write(prefix+'Rho_s/Rho_crit: '+str(halo.rhos)+end)
        outstream.write(prefix+'Virial Radius: '+str(halo.rvir)+' Mpc'+end)
        if halo.ucmh == 0:
            outstream.write(prefix+'Virial Concentration: '+str(halo.cvir)+end)
            outstream.write(prefix+'Core Radius: '+str(halo.rcore)+' Mpc'+end)
        outstream.write(prefix+'Luminosity Distance: '+str(halo.dl)+' Mpc'+end)
    outstream.write(prefix+'Angular Diameter Distance: '+str(halo.da)+' Mpc'+end)
    outstream.write(prefix+'Angular Observation Radius Per Arcmin: '+str(halo.rfarc)+' Mpc arcmin^-1'+end)
    if sim.theta > 0.0 and not sim.theta is None:  
        outstream.write(prefix+'Observation Radius for '+str(sim.theta)+' arcmin is '+str(halo.da*tan(sim.theta*2.90888e-4))+" Mpc"+end)
    if not sim.rintegrate is None:  
        outstream.write(prefix+'Observation Radius r_integrate '+str(sim.rintegrate)+" Mpc"+end)
    outstream.write(prefix+'======================================================'+end)
    outstream.write(prefix+'Gas Parameters: '+end)
    outstream.write(prefix+'======================================================'+end)
    if phys.ne_model == "flat":
        outstream.write(prefix+'Gas Distribution: '+"Flat Profile"+end)
        outstream.write(prefix+'Gas Central Density: '+str(phys.ne0)+' cm^-3'+end)
    elif(phys.ne_model == "powerlaw" or phys.ne_model == "pl"):
        outstream.write(prefix+'Gas Distribution: '+"Power-law profile"+end)
        outstream.write(prefix+'Gas Central Density: '+str(phys.ne0)+' cm^-3'+end)
        if phys.lb is None:
            outstream.write(prefix+'Scale radius: '+str(halo.rcore*1e3)+" kpc"+end)
        else:
            outstream.write(prefix+'Scale radius: '+str(phys.lb*1e3)+" kpc"+end)
        outstream.write(prefix+'PL Index: '+str(-1*phys.qe)+end)
    elif(phys.ne_model == "king"):
        outstream.write(prefix+'Gas Distribution: '+"King-type profile"+end)
        outstream.write(prefix+'Gas Central Density: '+str(phys.ne0)+' cm^-3'+end)
        if phys.lb is None:
            outstream.write(prefix+'Scale radius: '+str(halo.rcore*1e3)+" kpc"+end)
        else:
            outstream.write(prefix+'Scale radius: '+str(phys.lb*1e3)+" kpc"+end)
        outstream.write(prefix+'PL Index: '+str(-1*phys.qe)+end)
    elif(phys.ne_model == "exp"):
        outstream.write(prefix+'Gas Distribution: '+"Exponential"+end)
        outstream.write(prefix+'Gas Central Density: '+str(phys.ne0)+' cm^-3'+end)
        outstream.write(prefix+'Scale radius: '+str(halo.r_stellar_half_light*1e3)+" kpc"+end)

    if(write_mode == "flux"):
        outstream.write(prefix+'Gas Average Density (rvir): '+str(halo.neav)+' cm^-3'+end)
    outstream.write(prefix+'======================================================'+end)
    outstream.write(prefix+'Halo Substructure Parameters: '+end)
    outstream.write(prefix+'======================================================'+end)
    if(sim.sub_mode == "sc2006"):
        outstream.write(prefix+"Substructure Mode: Colafrancesco 2006"+end)
        outstream.write(prefix+"Substructure Fraction: "+str(halo.sub_frac)+end)
    elif(sim.sub_mode == "none" or sim.sub_mode is None):
        outstream.write(prefix+"Substructure Mode: No Substructure"+end)
    elif(sim.sub_mode == "prada"):
        outstream.write(prefix+"Substructure Mode: Sanchez-Conde & Prada 2013"+end)
        outstream.write(prefix+"Boost Factor: "+str(halo.boost)+end)
        outstream.write(prefix+"Synchrotron Boost Factor: "+str(halo.radio_boost)+end)
    outstream.write(prefix+'======================================================'+end)
    outstream.write(prefix+'Magnetic Field Parameters: '+end)
    outstream.write(prefix+'======================================================'+end)
    if(phys.b_flag == "flat"):
        outstream.write(prefix+'Magnetic Field Model: '+"Flat Profile"+end)
    elif(phys.b_flag == "powerlaw" or phys.b_flag == "pl"):
        outstream.write(prefix+'Magnetic Field Model: '+"Power-law profile"+end)
        outstream.write(prefix+'PL Index: '+str(-1*phys.qb*phys.qe)+end)
    elif(phys.b_flag == "follow_ne"):
        outstream.write(prefix+'Magnetic Field Model: '+"Following Gas Profile"+end)
        outstream.write(prefix+'PL Index on n_e: '+str(phys.qb)+end)
    elif(phys.b_flag == "equipartition"):
        outstream.write(prefix+'Magnetic Field Model: '+"Energy Equipartition with Gas"+end)
    elif(phys.b_flag == "sc2006"):
        outstream.write(prefix+'Magnetic Field Model: '+"Two-Parameter Coma Profile"+end)
        outstream.write(prefix+'Magnetic Field Scaling Radii: '+str(halo.rb1)+" Mpc "+str(halo.rb2)+" Mpc"+end)
    elif(phys.b_flag == "exp"):
        outstream.write(prefix+'Magnetic Field Model: '+"Exponential"+end)
        if phys.qb == 0.0:
            outstream.write(prefix+'Scale radius: '+str(halo.r_stellar_half_light*1e3)+" kpc"+end)
        else:
            outstream.write(prefix+'Scale radius: '+str(phys.qb*1e3)+" kpc"+end)
    elif(phys.b_flag == "m31"):
        outstream.write(prefix+'Magnetic Field Model: '+"M31"+end)
        outstream.write(prefix+'Scale radius r1: '+str(phys.qb*1e3)+" kpc"+end)
    elif(phys.b_flag == "m31exp"):
        outstream.write(prefix+'Magnetic Field Model: '+"M31 + Exponential after 14 kpc"+end)
        outstream.write(prefix+'Scale radius r1: '+str(phys.qb*1e3)+" kpc"+end)
    outstream.write(prefix+'Magnetic Field Strength Parameter: '+str(phys.b0)+' micro Gauss'+end)
    outstream.write(prefix+'Magnetic Field Average Strength (rvir): '+str(halo.bav)+" micro Gauss"+end)
    if(phys.diff == 0):
        outstream.write(prefix+'No Diffusion'+end)
    else:
        outstream.write(prefix+'Spatial Diffusion'+end)
        outstream.write(prefix+'Turbulence scale: '+str(phys.lc)+' kpc'+end)
        outstream.write(prefix+'Turbulence Index: '+str(phys.delta)+end)
        outstream.write(prefix+'Diffusion constant: '+str(phys.d0)+" cm^2 s^-1"+end)
    if not log is None:
        outstream.close()

def write_file(file_name,data,cols,index_row=0,append=False):
    """
    Write data to a file
        ---------------------------
        Parameters
        ---------------------------
        file_name - Required : output file name (String)
        data      - Required : 2D array-like, each row is written as data column in file
        cols      - Required : number of columns to write (int)
        index_row - Optional : number of rows before double line break (separates multiple 2D data sets) (int)
        append    - Optional : if True append to end of file_name (boolean)
        ---------------------------
        Output
        ---------------------------
        Writes to a file
    """
    try:
        if not append:
            outfile = open(file_name,"w")
        else:
            outfile = open(file_name,"a")
    except:
        fatal_error("I/O Error: Could not open "+file_name+" for writing")
    rows = len(data[0])
    #loop over how many rows must be written
    for r in range(0,rows):
        #write each column before inserting line break
        for c in range(0,cols):
            outfile.write(str(data[c][r]))
            if(c < cols-1):
                outfile.write(" ")
            else:
                outfile.write("\n")
        if(index_row != 0 and r%index_row == 0):
            outfile.write("\n")
            outfile.write("\n")
    outfile.close()

def write_electrons(file_id,electrons,r_set,E_set,me):
    """
    Write electron distribution to two specially formatted files
        ---------------------------
        Parameters
        ---------------------------
        file_id   - Required : output file name prefix (String)
        electrons - Required : 2D array-like electon distribution (len(E_set) x len(r_set)) [cm^-3]
        r_set     - Required : radial position values [Mpc]
        E_set     - Required : Lorentz gamma values []
        me        - electron mass [GeV]
        ---------------------------
        Output
        ---------------------------
        Files (file_id_electrons_mat.out) and (file_id_electrons.out)
        Writes to first file with electrons[i][j] with i being rows and j being columns
        Write to second file with  E_set[i]*me r_set[j] electrons[i][j] on each line 
    """
    try:
        eqout = open(file_id+"_electrons_mat.out","w")
        eqout2 = open(file_id+"_electrons.out","w")
    except:
        fatal_error("I/O Error: Could not open "+file_id+"_electrons.out"+" for writing")
    n = len(r_set)
    k = len(E_set)
    for i in range(0,k):
        for j in range(0,n):
            #eqout.write(str(i)+" "+str(j)+" "+str(electrons[i][j])+"\n")
            eqout.write(str(electrons[i][j])+" ")
            eqout2.write(str(E_set[i]*me)+" "+str(r_set[j])+" "+str(electrons[i][j])+"\n")
        eqout.write("\n")
        #eqout.write("\n")
        #eqout2.write("\n")
        #eqout2.write("\n")
    eqout.close()
    eqout2.close()
