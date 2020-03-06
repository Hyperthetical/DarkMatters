import sys,getopt,time,os
from numpy import *
from wimp_tools import tools,cosmology_env,simulation_env,physical_env,loop_env,halo_env #wimp handling package
from emm_tools import electron,radio,high_e,neutrino #emmisivity package
from multiprocessing import Pool
from os.path import join,isfile,isdir
from scipy.interpolate import interp1d
#==========================================
            #Preliminaries
#==========================================
#This code was written by Geoff Beck, based on IDL code by Paolo Marchegiani
#This project is made up of the following files
#
#dark_matters.py -> this file, it coordinates everything
#
#in the wimp_tools package the following are used:
#tools.py -> calculation tools, integration routines, simple/ubiquitous functions
#cosmology.py -> cosmology related calculation functions
#substructure.py -> methods for calculating spherically averaged substructure boost effects
#mass_function.py -> methods for computing mass functions,counts etc.
#ucmh.py -> routines for ultra-compact mini-halos
#
#in the emm_tools package:
#radio.py -> radio emmisivity, flux, surface brightness calculations
#high_e.py -> high energy (x-ray and gamma) emmisivity, flux, surface brightness calculations
#electron.py -> calculates electron equilibrium distributions, contains dedicated functions as well
#==========================================
            #Setup Instructions
#==========================================
#Parameters are set in a chosen file (here called 'input.data')
#to set a parameter add a line of the form:
#set parameter_name value
#where name and value are chosen appropriately
#simulation parameters:
#r_num (int) -> the number of spherical shells for radial integration -> accuracy can be achived with only 50 even for large halos
#gr_num (int) -> number of spherical shells for Green's function integration, must be large, recommened at least 500 
#nloop (int) [optional] -> number of masses sampled for calculations of mass functions
#f_num (int) -> the number of frequency bins, I recommend 41 - 51 for 5 orders of magnitude range 
#flim (float,float - MHz) -> minimum and maximum frequencies sampled
#nu_sb (float - MHz) -> the frequency for calculating the surface brightness at 
#theta (float) [optional] -> the angular radius for flux integration in arcmin, used only when not calculating virial fluxes
#input_dir (string) -> input directory path where input data files are stored
#
#wimp parameters:
#wimp_mass (float - GeV) -> mass of WIMP particle
#channel (string) -> dominant annihilation channel
#
#Halo parameters: 
#z (float) -> halo redshift
#mvir (float - solar masses)[1] -> halo virial mass
#rvir (float - mpc)[1] -> halo virial radius
#cvir (float) [optional] -> specify virial concentration yourself rather than have it computed
#r_core (float - mpc) [optional] -> specify halo characteristic radius yourself rather than have it computed
#profile (options: nfw,einasto,isothermal, or burkert) -> halo profile 
#alpha (float) [optional] -> alpha constant for einasto profile, is required if using einasto profile mode
#ne (float - cm^-3) -> thermal plasma central density
#qe (float) -> thermal scaling exponent
#sub_mode (options: none, sc2006, prada) -> substructure modes
#fs (float) [optional] -> substructure mass fraction, used only with sc2006 mode
#[1] = must specify at least one of these
#
#Magnetic field and turbulence: 
#B (float - muG) -> magnetic field average/central strength
#b_flag (options: 0,1,2,3,4) -> field models (constant,scales with radius exponent qb*qe,scales with sqrt(ne), gas energy normalised, two angle model
#r_field (float,float - arcsec) [optional] -> two angles for b_flag 4 mode
#qb (float) -> magnetic field scaling exponent 
#d (float- kpc) -> minimum homogenuity scale 
#delta (float) -> turbulence spectral index
#diff (options: 1 or 0) -> turns diffusion effects on or off
#
#==========================================
            #Run Instructions
#==========================================
#to run use the command "python3 dark_matters.py input.data"
#
#==========================================
            #Output Instructions
#==========================================
#Output has several forms
#Flux densities are output in a file that is named chmx_mh_za_bf_dn_prof_fsk_(arcmin/vir)flux.out
#here prof is the profile type,  n is the diffusuion flag value and a is the redshift. 
#ch is the channel, mx is the wimp mass (int), mh is the log10 halo mass, k is the sub-halo mass fraction
#The file has 3 columns: frequency(MHz) Flux(Jy) Flux(erg cm^-2 s^-1)
#Surface Brightnesses are output in a file that is named in the same fashion but terminates in sb.out rather than the (arcmin/vir)flux.out
#the file has 2 columns: Angle from halo center (arcmin) Surface Brightness (Jy arcmin^-2)
#Mass loop calculations are output to file: loop_chmx_bf_za_dn_prof_fsk_uMHz.out
#u is the frequency at which the calculations were performed (int)
#5 columns: halo mass(solar masses) luminosity(erg s^-1 Hz^-1) SB(Jy arcmin^-2) Virial Flux(Jy) Arcmin Flux (Jy)
#NB - no output includes the cross section - you must multiply by sig_v where appropriate when plotting

MHzToGeV = 1e6*4.136e-15*1e-9  #conversion factor
#specdir = "/home/geoff/Coding/Wimps/data_input/susy/"

def process_file(command_file,phys,sim,halo,cos_env,loop):
    """
    Reads a file of commands in order to execute them line by line
        ---------------------------
        Parameters
        ---------------------------
        command_file - Required : command file path
        phys         - Required : physical environment (phys_env)
        sim          - Required : simulation environment (simulation_env)
        halo         - Required : halo environment(halo_env)
        cos_env      - Required : cosmology environment (cosmology_env)
        loop         - Required : loop environment (loop_env)
        ---------------------------
        Output
        ---------------------------
        None
    """
    try:
        infile = open(command_file,"r")
    except:
        tools.fatal_error("File "+command_file+" is not a valid file path")
    lines = []
    for line in infile:
        lines.append(line.rstrip())
    infile.close()
    for line in lines:
        process_command(line,phys,sim,halo,cos_env,loop)

def set_theta_flag(s):
    """
    Works out an integration region from a flux calculation command
        ---------------------------
        Parameters
        ---------------------------
        s - Required : a list of strings (array-like, String)
        ---------------------------
        Output
        ---------------------------
        String with 'full', 'r_integrate', or 'theta' value
    """
    try:
        if s[2] == "full":
            theta_flag = "full"
        elif s[2] == "r_integrate":
            theta_flag = "r_integrate"
        else:
            theta_flag = "theta"
    except:
        theta_flag = "full"
    return theta_flag


def process_command(command,phys,sim,halo,cos_env,loop):
    """
    Extracts and executes the command in a line of a command file
        ---------------------------
        Parameters
        ---------------------------
        command - Required : command line to be executed
        phys    - Required : physical environment (phys_env)
        sim     - Required : simulation environment (simulation_env)
        halo    - Required : halo environment(halo_env)
        cos_env - Required : cosmology environment (cosmology_env)
        loop    - Required : loop environment (loop_env)
        ---------------------------
        Output
        ---------------------------
        None - evaluates script commands
    """
    calc_dict = {"rflux":"Radio flux","hflux":"High-frequency flux","flux":"All flux","jflux":"Gamma-ray flux from Jfactor","gflux":"Gamma-ray flux","nuflux":"Neutrino flux","nu_jflux":"Neutrino flux from J-factor","sb":"Radio surface brightness","gamma_sb":"gamma-ray surface brightness","emm":"Emmissivity","loop":"Radio loop","hloop":"High energy loop","electrons":"Electron distributions"}
    if "#" in command.strip() and not command.strip().startswith("#"):
        s = command.split("#")[0].split()
    elif not command.strip() == "":
        s = command.split()
    if command.strip().startswith("#") or command.strip() == "":
        pass
    elif command.strip().lower() == "calc help":
        print("The calc command is used as follows: calc mode region")
        print("region -> full, theta, r_integrate")
        print("mode can take the values")
        for c in calc_dict:
            print(c+" calculates "+calc_dict[c])
    elif s[0] in ["batch","b","c","calc"]:
        try:
            calc_str = s[1]
            try:
                t_str = s[2]
            except:
                t_str = "full"
        except:
            calc_str = "flux"
            t_str = "full"
        #print(calc_str)
        batch_calc = 0
        batch_time = time.time()
        for m in sim.mx_set:
            phys.clear_spectra()
            batch_calc += 1
            print("=========================================================")
            print("Beginning new Dark Matters calculation")
            print("=========================================================")
            try:
                print("Calculation Task Given: "+calc_dict[s[1]])
            except IndexError:
                tools.fatal_error("calc must be supplied with an option: "+str(calc_set))
            except KeyError:
                tools.fatal_error("calc must be supplied with a valid option from: "+str(calc_set))
            process_command("set wimp_mass "+str(m),phys,sim,halo,cos_env,loop)
            if sim.specdir is None:
                tools.fatal_error("Please specify a directory for input spectra with 'set input_spectra_directory #' command")
            if(not halo.check_self()):
                tools.fatal_error("halo data underspecified")
            if(not phys.check_self()):
                tools.fatal_error("physical environment data underspecified")
            if(not sim.check_self()):
                tools.fatal_error("simulation data underspecified")
            print("All input data consistency checks complete")
            if "nu" in calc_str:
                neutrino_spectrum(sim.specdir,phys,sim,sim.nu_flavour,mode=halo.mode)
            if(not phys.check_particles()):
                tools.fatal_error("particle data underspecified")
            gather_spectrum(sim.specdir,phys,sim,mode=halo.mode)
            process_command("setup halo",phys,sim,halo,cos_env,loop)
            #print("c "+calc_str+" "+t_st)
            process_command("internal_c "+calc_str+" "+t_str,phys,sim,halo,cos_env,loop)
        print("Batch Time: "+str(time.time()-batch_time)+" s")
        print("Batch Time per Calculation: "+str((time.time()-batch_time)/batch_calc)+" s")
    elif(s[0] == "internal_calc" or s[0] == "internal_c"):
        calc_set = ["rflux","hflux","flux","jflux","gflux","nuflux","nu_jflux","sb","gamma_sb","emm","loop","hloop","electrons"]
        diff_calc_set = ["rflux","hflux","flux","sb","emm","loop","hloop","electrons"]
        try:
            str(s[1])
        except IndexError:    
            tools.fatal_error("calc must be supplied with an option: "+str(calc_set))
        if not str(s[1]) in calc_set:
            tools.fatal_error("calc must be supplied with a valid option: "+str(calc_set))
        if(not halo.check_self()):
            tools.fatal_error("halo data underspecified")
        if(not phys.check_self()):
            tools.fatal_error("physical environment data underspecified")
        if(not sim.check_self()):
            tools.fatal_error("simulation data underspecified")
        if not halo.name is None:
            nameFirst = True
        else:
            nameFirst = False
        #if "nu" in s[1]:
        #    neutrino_spectrum(sim.specdir,phys,sim.nu_flavour,mode=halo.mode)
        if halo.J_flag != 0 and not "jflux" in s[1]:
            print("=========================================================")
            print("Calculating Flux Normalisation to Match Given J-Factor")
            print("=========================================================")
            hfmax = phys.mx/(1e6*4.136e-15*1e-9) #MHz
            hsim = simulation_env(n=sim.n,ngr=sim.ngr,num=20,fmin=1e-3*hfmax,fmax=0.1*hfmax,theta=sim.theta,nu_sb=sim.nu_sb)
            hsim.sample_f()
            theta_flag = set_theta_flag(s)
            get_h_flux(halo,hsim,0,theta_flag,suppress_output=True)
            if theta_flag == "full":
                gflux = halo.he_virflux
            else:
                gflux = halo.he_arcflux
            jflux = high_e.gamma_from_j(halo,phys,hsim)*4.14e-24*1.6e20
            fRatio = sum(jflux/gflux)/len(jflux)
            sim.jnormed = True
            print("Normalisation factor: "+str(fRatio))
            halo.he_arcflux = None;halo.he_virflux=None
        else:
            fRatio = 1.0
        if s[1] in diff_calc_set:
            theta_flag = set_theta_flag(s)
            if theta_flag == "full":
                rmax = halo.rvir
            elif theta_flag == "theta":
                rmax = halo.da*tan(sim.theta*2.90888e-4)
            elif theta_flag == "r_integrate":
                rmax = sim.rintegrate
            else:
                rmax = halo.rvir
                print("Warning: region flag "+s[2]+" in command "+command+" not recognised, defaulting to \'full\'")
            halo.physical_averages(rmax)
        #print("fRatio",fRatio)
        if(s[1] == "rflux"):
            if(halo.radio_emm == None):
                if(halo.electrons == None):
                    time_start = time.time()
                    get_electrons(halo,phys,sim)
                    print("Time taken: "+str(time.time()-time_start)+" s")
                    time_start = time.time()
                    get_emm(halo,phys,sim)
                    print("Time taken: "+str(time.time()-time_start)+" s")
                else:
                    time_start = time.time()
                    get_emm(halo,phys,sim)
                    print("Time taken: "+str(time.time()-time_start)+" s")
            theta_flag = set_theta_flag(s)
            if not halo.name is None:
                nameFirst = True
            else:
                nameFirst = False
            get_flux(halo,sim,theta_flag)
            if theta_flag == "theta":
                fluxout = get_file_id(sim,phys,cos_env,halo,nameFirst)+"_radio_"+str(sim.theta)+"arcminflux.out"
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                erg = halo.radio_arcflux*sim.f_sample*1e-17
                write = [];write.append(sim.f_sample);write.append(halo.radio_arcflux);write.append(erg)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
            elif theta_flag == "r_integrate":
                fluxout = get_file_id(sim,phys,cos_env,halo,nameFirst)+"_radio_"+str(sim.rintegrate)+"mpcflux.out"
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                erg = halo.radio_arcflux*sim.f_sample*1e-17
                write = [];write.append(sim.f_sample);write.append(halo.radio_arcflux);write.append(erg)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
            elif theta_flag == "full":
                fluxout = get_file_id(sim,phys,cos_env,halo,nameFirst)+"_radio_virflux.out"
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                erg = halo.radio_virflux*sim.f_sample*1e-17
                cm21 = 1.4e3
                if sim.num > 1:
                    fluxint = interp1d(sim.f_sample,erg)
                    print("Flux at 1.4 GHz: "+str(fluxint(cm21)))
                write = [];write.append(sim.f_sample);write.append(halo.radio_virflux);write.append(erg)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
        elif(s[1] == "flux"):
            if halo.electrons == None:
                time_start = time.time()
                get_electrons(halo,phys,sim)
                print("Time taken: "+str(time.time()-time_start)+" s")
            #tools.write_electrons("test",halo.electrons,halo.r_sample[0],phys.spectrum[0],0.511e-3)
            if halo.radio_emm == None:
                time_start = time.time()
                get_emm(halo,phys,sim)
                print("Time taken: "+str(time.time()-time_start)+" s")
            if halo.he_emm == None:
                    time_start = time.time()
                    get_h_emm(halo,phys,sim)
                    print("Time taken: "+str(time.time()-time_start)+" s")
            ##cosmology.jfactor(halo)
            theta_flag = set_theta_flag(s)
            get_h_flux(halo,sim,1,theta_flag)
            get_flux(halo,sim,theta_flag)
            get_multi_flux(halo)
            if theta_flag == "theta":
                fluxout = get_file_id(sim,phys,cos_env,halo,nameFirst)+"_multi_"+str(int(sim.theta))+"arcminflux.out"
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                erg = halo.multi_arcflux*sim.f_sample*1e-17*fRatio
                write = [];write.append(sim.f_sample);write.append(halo.multi_arcflux);write.append(erg)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
            elif theta_flag == "r_integrate":
                fluxout = get_file_id(sim,phys,cos_env,halo,nameFirst)+"_multi_"+str(sim.rintegrate)+"mpcflux.out"
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                erg = halo.multi_arcflux*sim.f_sample*1e-17*fRatio
                write = [];write.append(sim.f_sample);write.append(halo.multi_arcflux);write.append(erg)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
            elif theta_flag == "full":
                fluxout = get_file_id(sim,phys,cos_env,halo,nameFirst)+"_multi_virflux.out"
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                erg = halo.multi_virflux*sim.f_sample*1e-17*fRatio
                write = [];write.append(sim.f_sample);write.append(halo.multi_virflux);write.append(erg)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
        elif(s[1] == "hflux"):
            #tools.write(None,"flux",sim,phys,cos_env,halo)
            if(halo.he_emm == None):
                if(halo.electrons == None):
                    time_start = time.time()
                    get_electrons(halo,phys,sim)
                    print("Time taken: "+str(time.time()-time_start)+" s")
                    time_start = time.time()
                    get_h_emm(halo,phys,sim)
                    print("Time taken: "+str(time.time()-time_start)+" s")
                else:
                    time_start = time.time()
                    get_h_emm(halo,phys,sim)
                    print("Time taken: "+str(time.time()-time_start)+" s")
            #cosmology.jfactor(halo)
            theta_flag = set_theta_flag(s)
            get_h_flux(halo,sim,1,theta_flag)
            if theta_flag == "theta":
                fluxout = get_file_id(sim,phys,cos_env,halo,nameFirst)+"_he_"+str(sim.theta)+"arcminflux.out"
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                erg = halo.he_arcflux*sim.f_sample*1e-17*fRatio
                write = [];write.append(sim.f_sample);write.append(halo.he_arcflux);write.append(erg)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
            elif theta_flag == "r_integrate":
                fluxout = get_file_id(sim,phys,cos_env,halo,nameFirst)+"_he_"+str(sim.rintegrate)+"mpcflux.out"
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                erg = halo.he_arcflux*sim.f_sample*1e-17*fRatio
                write = [];write.append(sim.f_sample);write.append(halo.he_arcflux);write.append(erg)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
            elif theta_flag == "full":
                fluxout = get_file_id(sim,phys,cos_env,halo,nameFirst)+"_he_virflux.out"
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                erg = halo.he_virflux*sim.f_sample*1e-17*fRatio
                write = [];write.append(sim.f_sample);write.append(halo.he_virflux);write.append(erg)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
        elif(s[1] == "gflux"):
            #tools.write(None,"flux",sim,phys,cos_env,halo)
            high_e.gamma_source(halo,phys,sim)
            #cosmology.jfactor(halo)
            theta_flag = set_theta_flag(s)
            get_h_flux(halo,sim,0,theta_flag)
            if theta_flag == "theta":
                fluxout = halo.name+"_"+short_id(sim,phys,halo)+"_"+halo.profile+"_gamma_"+str(sim.theta)+"arcminflux.out"
                erg = halo.he_arcflux*sim.f_sample*1e-17
                write = [];write.append(sim.f_sample);write.append(halo.he_arcflux);write.append(erg)
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
            elif theta_flag == "r_integrate":
                fluxout = halo.name+"_"+short_id(sim,phys,halo)+"_"+halo.profile+"_gamma_"+str(sim.rintegrate)+"mpcflux.out"
                erg = halo.he_arcflux*sim.f_sample*1e-17
                write = [];write.append(sim.f_sample);write.append(halo.he_arcflux);write.append(erg)
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
            elif theta_flag == "full":
                fluxout = halo.name+"_"+short_id(sim,phys,halo)+"_"+halo.profile+"_gamma_virflux.out"
                erg = halo.he_virflux*sim.f_sample*1e-17
                write = [];write.append(sim.f_sample);write.append(halo.he_virflux);write.append(erg)
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
        elif(s[1] == "nuflux"):
            #tools.write(None,"flux",sim,phys,cos_env,halo)
            get_nu_emm(halo,phys,sim)
            if phys.nu_flavour == "mu":
                nu_str = "_nu_mu_"
            elif phys.nu_flavour == "e":
                nu_str = "_nu_e_"
            elif phys.nu_flavour == "tau":
                nu_str = "_nu_tau_"
            else:
                nu_str = "_nu_"
            theta_flag = set_theta_flag(s)
            get_nu_flux(halo,sim,0,theta_flag)
            if theta_flag == "theta":
                fluxout = halo.name+"_"+short_id(sim,phys,halo)+"_"+halo.profile+nu_str+str(sim.theta)+"arcminflux.out"
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                erg = halo.nu_arcflux*sim.f_sample*1e-17
                write = [];write.append(sim.f_sample*MHzToGeV);write.append(halo.nu_arcflux/1.6e20/4.14e-24);write.append(erg)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
            elif theta_flag == "r_integrate":
                fluxout = halo.name+"_"+short_id(sim,phys,halo)+"_"+halo.profile+nu_str+str(sim.rintegrate)+"mpcflux.out"
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                erg = halo.nu_arcflux*sim.f_sample*1e-17
                write = [];write.append(sim.f_sample*MHzToGeV);write.append(halo.nu_arcflux/1.6e20/4.14e-24);write.append(erg)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
            if theta_flag == "full":
                fluxout = halo.name+""+short_id(sim,phys,halo)+"_"+halo.profile+nu_str+"virflux.out"
                tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
                erg = halo.nu_virflux*sim.f_sample*1e-17
                write = [];write.append(sim.f_sample*MHzToGeV);write.append(halo.nu_virflux/1.6e20/4.14e-24);write.append(erg)
                tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
        elif(s[1] == "jflux"):
            halo.he_virflux = None
            halo.he_virflux = high_e.gamma_from_j(halo,phys,sim)
            if halo.mode == "ann":
                fluxout = halo.name+"_"+short_id(sim,phys,halo)+"_he_jflux.out"
            else:
                fluxout = halo.name+"_decay_"+short_id(sim,phys,halo)+"_he_jflux.out"
            erg = halo.he_virflux*sim.f_sample*1e-17*4.14e-24*1.6e20
            write = [];write.append(sim.f_sample);write.append(halo.he_virflux);write.append(erg)
            tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
            tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
            fluxout = None;write = None
        elif(s[1] == "nu_jflux"):
            #tools.write(None,"flux",sim,phys,cos_env,halo)
            halo.he_virflux = None
            print("J-Factor: "+str(halo.J))
            halo.nu_virflux = neutrino.nu_from_j(halo,phys,sim)
            if phys.nu_flavour == "mu":
                nu_str = "_nu_mu_"
            elif phys.nu_flavour == "e":
                nu_str = "_nu_e_"
            elif phys.nu_flavour == "tau":
                nu_str = "_nu_tau_"
            else:
                nu_str = "_nu_"
            if halo.mode == "ann":
                fluxout = halo.name+"_"+short_id(sim,phys,halo)+nu_str+"jflux.out"
            else:
                fluxout = halo.name+"_decay_"+short_id(sim,phys,halo)+nu_str+"jflux.out"
            erg = halo.nu_virflux*sim.f_sample*1e-17*4.14e-24*1.6e20 #cm^-2 s^-1 * Jy -> erg cm^-2 s^-1 * h {GeV s} * GeV cm^-2 -> Jy * 
            write = [];write.append(sim.f_sample*MHzToGeV);write.append(halo.nu_virflux);write.append(erg)
            tools.write(join(sim.out_dir,fluxout),"flux",sim,phys,cos_env,halo)
            tools.write_file(join(sim.out_dir,fluxout),write,3,append=True)
            fluxout = None;write = None
        elif(s[1] == "sb"):
            if(halo.electrons == None):
                time_start = time.time()
                get_electrons(halo,phys,sim)
                print("Time taken: "+str(time.time()-time_start)+" s")
            if(halo.radio_emm_nu == None or halo.gamma_emm_nu == None or halo.he_emm_nu == None):
                time_start = time.time()
                get_emm_nu(halo,phys,sim)
                print("Time taken: "+str(time.time()-time_start)+" s")
            time_start = time.time()
            sync_sb = get_sync_sb(halo,sim)
            print("Process Complete")
            xray_sb = get_xray_sb(halo,sim)
            print("Process Complete")
            gamma_sb = get_gamma_sb(halo,sim)
            print("Process Complete")
            print("Time taken for 3 Surface-Brightnesses: "+str(time.time()-time_start)+" s")
            sbout = get_file_id(sim,phys,cos_env,halo,nameFirst)+"_sb.out"
            radians_per_arcmin = 2.909e-4 #conversion factor
            theta_sample = arctan(halo.r_sample[0]/halo.da)/radians_per_arcmin
            write = [];write.append(theta_sample);write.append(sync_sb);write.append(xray_sb);write.append(gamma_sb)
            tools.write(sbout,"flux",sim,phys,cos_env,halo)
            tools.write_file(sbout,write,len(write),append=True)
        elif(s[1] == "gamma_sb"):
            if(halo.gamma_emm_nu == None):
                time_start = time.time()
                get_gamma_emm_nu(halo,phys,sim)
                print("Time taken: "+str(time.time()-time_start)+" s")
            if not halo.name is None:
                nameFirst = True
            else:
                nameFirst = False
            time_start = time.time()
            gamma_sb = get_gamma_sb(halo,sim)
            print("Process Complete")
            print("Time taken for Surface-Brightness: "+str(time.time()-time_start)+" s")
            print(halo.J_flag,halo.da)
            if halo.J_flag == 1: 
                sbout = halo.tag+"_"+short_id(sim,phys,halo)+"_gsb.out"
            else:
                sbout = get_file_id(sim,phys,cos_env,halo,nameFirst)+"_gsb.out"
            radians_per_arcmin = 2.909e-4 #conversion factor
            theta_sample = arctan(halo.r_sample[0]/halo.da)/radians_per_arcmin
            write = [];write.append(theta_sample);write.append(gamma_sb)
            tools.write(sbout,"flux",sim,phys,cos_env,halo)
            tools.write_file(sbout,write,len(write),append=True)
        elif(s[1] == "emm"):
            if(halo.electrons == None):
                time_start = time.time()
                get_electrons(halo,phys,sim)
                print("Time taken: "+str(time.time()-time_start)+" s")
            else:
                time_start = time.time()
                get_emm(halo,phys,sim)
                print("Time taken: "+str(time.time()-time_start)+" s")
        elif(s[1] == "electrons"):
            time_start = time.time()
            get_electrons(halo,phys,sim)
            print("Time taken: "+str(time.time()-time_start)+" s")
        elif(s[1] == "loop"):
            tools.write(None,"loop",sim,phys,cos_env,halo)
            loop_sim = sim
            loop.sim = loop_sim;loop.cosm = cos_env;loop.phys = phys
            loop.sub_frac = halo.sub_frac;loop.alpha = halo.alpha
            loop.dm = halo.dm
            wimp_str = loop.phys.channel+str(int(loop.phys.mx))
            field_str = "b"+str(int(loop.phys.b0))+"q"+str(loop.phys.qb)
            if(loop.dm == 1):
                dm_str = "nfw"
            elif(loop.dm == -1):
                dm_str = "ein"+str(loop.alpha)
            else:
                dm_str = "dm"+str(loop.dm)
            loopout = "loop"+"_"+wimp_str+"_"+field_str+"_"+"d"+str(loop.phys.diff)+"_"+dm_str+"_"+"fs"+str(loop.sub_frac)+".out"
            outf = open(loopout,"w")
            zsample = logspace(log10(loop.zmin),log10(loop.zmax),num=loop.zn)
            
            for j in range(0,loop.zn):
                print("Preparing Loop: "+str(j+1)+"/"+str(loop.zn)+", z: "+str(zsample[j]))
                time_start = time.time()
                loop.setup(z_index=j)
                argset = []
                for i in range(0,loop.mn):
                    argset.append([loop.halos[i],loop.sim,loop.phys_set[i]])
                pool = Pool(processes=4)
                pool.map(loop_mass,argset)
                #for i in range(0,loop.mn):
                #    print "Executing Sub-Loop: "+str(i+1)+"/"+str(loop.mn)+", mass: "+str(loop.halos[i].mvir)+", z: "+str(loop.halos[i].z)
                #    get_electrons(loop.halos[i],loop.sim)
                #    get_emm(loop.halos[i],loop.phys_set[i],loop.sim)
                #    get_flux(loop.halos[i],loop.sim,"full")
                for i in range(0,loop.mn):
                    fstr = ""
                    for k in range(0,sim.num):
                        fstr += " "+str(loop.halos[i].virflux[k])    
                    outf.write(str(loop.z_sample[j])+" "+str(loop.m_sample[i])+fstr+"\n")
                #fw.append(zf)
                print("Time Taken: "+str(time.time()-time_start))
            #write=[];write.append(loop.m_sample);write.append(fw)
            #tools.write_file(loopout[0],write,2)
            #for i in range(0,loop.zn):
            #    for j in range(0,loop.mn):
            #        fstr = ""
            #        for k in range(0,sim.num):
            #            fstr += " "+str(fw[i][j][k])    
                    #outf.write(str(loop.z_sample[i])+" "+str(loop.m_sample[j])+fstr+"\n")
            outf.close()
        elif(s[1] == "hloop"):
            tools.write(None,"loop",sim,phys,cos_env,halo)
            loop_sim = sim
            loop.sim = loop_sim;loop.cosm = cos_env;loop.phys = phys
            loop.sub_frac = halo.sub_frac;loop.alpha = halo.alpha
            loop.dm = halo.dm
            wimp_str = loop.phys.channel+str(int(loop.phys.mx))
            field_str = "b"+str(int(loop.phys.b0))+"q"+str(loop.phys.qb)
            if(loop.dm == 1):
                dm_str = "nfw"
            elif(loop.dm == -1):
                dm_str = "ein"+str(loop.alpha)
            else:
                dm_str = "dm"+str(loop.dm)
            loopout = "hloop"+"_"+wimp_str+"_"+field_str+"_"+"d"+str(loop.phys.diff)+"_"+dm_str+"_"+"fs"+str(loop.sub_frac)+".out"
            outf = open(loopout,"w")
            for j in range(0,loop.zn):
                print("Preparing Loop: "+str(j)+"/"+str(loop.zn))
                time_start = time.time()
                loop.setup(z_index=j)
                gamma = 1 #gamma only flag 
                for i in range(0,loop.mn):
                    print("Executing Sub-Loop: "+str(i)+"/"+str(loop.mn))
                    if(gamma == 1):
                        get_electrons(loop.halos[i],loop.phys_set[i],loop.sim)
                        get_h_emm(loop.halos[i],loop.phys_set[i],loop.sim)
                    get_h_flux(loop.halos[i],loop.sim,gamma,"full")
                    #zf.append(loop.halos[i].he_virflux)
                    fstr = ""
                    for k in range(0,sim.num):
                        fstr += " "+str(loop.halos[i].he_virflux[k])    
                    print(fstr)
                    outf.write(str(loop.z_sample[j])+" "+str(loop.m_sample[i])+fstr+"\n")
                #fw.append(zf)
                print("Time Taken: "+str(time.time()-time_start))
            #write=[];write.append(loop.m_sample);write.append(fw)
            #tools.write_file(loopout[0],write,2)
            #for i in range(0,loop.zn):
            #    for j in range(0,loop.mn):
            #        fstr = ""
            #        for k in range(0,sim.num):
            #            fstr += " "+str(fw[i][j][k])    
                    #outf.write(str(loop.z_sample[i])+" "+str(loop.m_sample[j])+fstr+"\n")
            outf.close()
        else:
            tools.fatal_error("unrecognised calculation option: "+s[1])
    elif(s[0] == "setup"):
        try:
            if(s[1] == "halo"):
                if(not halo.setup(sim,phys,cos_env)):
                    tools.fatal_error("halo setup error")
                log = get_file_id(sim,phys,cos_env,halo,False)+".log"
                if sim.log_mode == 1:
                    tools.write(log,"flux",sim,phys,cos_env,halo)
                tools.write(None,"flux",sim,phys,cos_env,halo)
        except IndexError:
            tools.fatal_error("setup must be followed by a valid command")
    elif(s[0] == "set"):
        process_set(command,phys,sim,halo,cos_env,loop)
    elif(s[0] == "help" or s[0] == "h"):
        try:
            tools.help_menu(s[1])
        except IndexError:
            tools.help_menu()
    elif(s[0] == "load" or s[0] == "l"):
        try:
            open(s[1],"r")
            process_file(s[1],phys,sim,halo,cos_env,loop)
        except IndexError:
            tools.fatal_error("the 'load' command requires a valid input file path to be supplied")
    elif(s[0] == "exit" or s[0] == "quit" or s[0] == "q"):
        sys.exit(2)
    elif(s[0] == "show"):
        try:
            if(s[1] == "log"):
                log = get_file_id(sim,phys,cos_env,halo,False)+".log"
                tools.write(log,"flux",sim,phys,cos_env,halo)
        except IndexError:
            tools.fatal_error("show must be followed by a valid command")
    else:
        print("Invalid Command: "+command)

def process_set(line,phys,sim,halo,cos_env,loop):
    """
    Sets a parameter in one of the environment objects
        ---------------------------
        Parameters
        ---------------------------
        line     - Required : command file line starting with 'set'
        phys     - Required : physical environment (phys_env)
        sim      - Required : simulation environment (simulation_env)
        halo     - Required : halo environment(halo_env)
        cos_env  - Required : cosmology environment (cosmology_env)
        loop     - Required : loop environment (loop_env)
        ---------------------------
        Output
        ---------------------------
        None - assigns values to environment attributes 
    """
    hsp_commands = {"rvir":[halo,"float","rvir","(mpc units)"],"mvir":[halo,"float","mvir","(solar mass units)"],"r_integrate":[sim,"float","rintegrate","(Mpc units)"],"rcore":[halo,"float","rcore","(mpc units)"],"r_stellar_half_light":[halo,"float","r_stellar_half_light","(mpc units)"],"qb":[phys,"float","qb","(function varies by profile)"],"b_model":[phys,"string","b_flag","(chosen magnetic field model)"],"B":[phys,"float","b0","(micro Gauss units)"],"particle_model":[phys,"string","particle_model","(name for particle physics model)"],"ne_model":[phys,"string","ne_model","(chosen electron model)"],"ne":[phys,"float","ne0","(cm^-3 units)"],"qe":[phys,"float","qe","(function varies by ne model)"],"jflag":[halo,"int","J_flag","(jfactor norm flag)"],"btag":[phys,"string","btag","(magnetic field label)"],"theta":[sim,"float","theta","(arcminute units)"],"profile":[halo,"string","profile","(halo density profile)"],"input_spectra_directory":[sim,"string","specdir","(directory with annihilation channel input spectra)"],"alpha":[halo,"float","alpha","(Einasto alpha)"],"f_num":[sim,"int","num","(number of frequency samples)"],"r_num":[sim,"int","n","(number of r samples)"],"gr_num":[sim,"int","ngr","(r samples in Green's functions)"],"e_bins":[sim,"int","e_bins","(number of energy bins for input spectra)"],"submode":[sim,"string","sub_mode","(substructure model)"],"diff":[phys,"int","diff","diffusion flag 1 or 0)"],"name":[halo,"string","name","(halo name)"],"d":[phys,"float","lc","(units kpc)"],"dist":[halo,"float","dl","(units mpc)"],"z":[halo,"float","z","(redshift)"],"nu_flavour":[sim,"string","nu_flavour","(neutrino flavour)"],"channel":[phys,"stringarray","channel","(annihilation channel)"],"branching":[phys,"floatarray","branching","(set of branching ratios in same order as channels, separated by spaces)"],"rhos":[halo,"float","rhos","(unitless)"],"rho0":[halo,"float","rho0","(msol mpc^-3)"],"mx_set":[sim,"floatarray","mx_set","(set of WIMP masses in GeV separated by spaces)"],"flim":[sim,"floatarray","flim","(min and max frequencies in MHz separated by white space)"],"ucmh":[halo,"int","ucmh","(ucmh flag 1 or 0)"],"sub_frac":[halo,"float","sub_frac","(mass fraction of sub-halos)"],"jfactor":[halo,"float","J","(units of GeV^2 cm^-5)"],"cvir":[halo,"float","cvir","(virial concentration)"],"t_star_formation":[halo,"float","t_sf","(star formation time in seconds)"],"b_average":[halo,"float","bav","(micro Gauss)"],"ne_average":[halo,"float","neav","(cm^-3)"],"wimp_mode":[halo,"string","mode","(ann or decay)"],"radio_boost":[sim,"int","radio_boost_flag","(1 or 0)"],"frequency":[sim,"float","nu_sb","(MHz)"],"delta":[phys,"float","delta","(turbulence index)"],"wimp_mass":[phys,"float","mx","(WIMP mass in GeV)"],"output_log":[sim,"int","log_mode","(1 or 0)"],"electrons_from_c":[],"radio_emm_from_c":[],"ne_scale":[phys,"float","lb","(ne scale length Mpc)"]}

    cosmo_commands = {"omega_m":[cos_env,"float","w_m","(matter fraction)"],'omega_lambda':[cos_env,"float","w_l","(cosmological constant fraction)"],'ps_index':[cos_env,"float","n","(matter power spectrum index)"],'curvature':[cos_env,"string","universe","(flat or or curved)"],'omega_b':[cos_env,"float","w_b","(baryon fraction)"],'omega_dm':[cos_env,"float","w_dm","(DM fraction)"],'sigma_8':[cos_env,"float","sigma_8","(power spectrum normalisation)"], 'omega_nu':[cos_env,"float","w_nu","(neutrino fraction)"],'N_nu':[cos_env,"float","N_nu","(neutrino number)"]}

    loop_commands = {"nloop":[loop,"int","mn","(number of mass samples in loop)"]}

    hsp2_commands = {"diffusion_constant":[phys,"float","d0","(diffusion constant (cm^2 s^-1))"],"isrf":[phys,"int","ISRF","(inter-sellar radiation field flag (1 or 0))"],"output_directory":[sim,"string","out_dir","(output directory)"],"nfw_index":[halo,"float","gnfw_gamma","(gnfw gamma index)"]}

    commands = {**hsp_commands,**cosmo_commands}
    commands = {**commands,**loop_commands,**hsp2_commands}

    

    sline = line.rstrip().split("#")[0]
    s = sline.rstrip().split()
    if len(s) < 2:
        tools.fatal_error("Command error: "+line+"\nYou need to set something")
    if not s[1] in commands and s[1] != "help":
        tools.fatal_error("Command error: "+s[1]+" does not exist")
    if line.strip().lower() == "set help":
        enumerate_set_commands(commands)
    elif s[1] == "electrons_from_c":
        try:
            yl = s[2]
            if "T" in yl or "t" in yl:
                sim.electrons_from_c = True
            else:
                sim.electrons_from_c = False
        except IndexError:
            tools.fatal_error("electrons_from_c requires true or false as first argument")
        if sim.electrons_from_c:
            try:
                yl = s[3]
                if not isfile(yl):
                    tools.fatal_error("Cannot find the c executable: "+yl+"\nelectrons_from_c requires true or false as first argument and the path to the c executable as the second")
                sim.exec_electron_c = yl
            except IndexError:
                tools.fatal_error("electrons_from_c requires true or false as first argument and the path to the c executable as the second")
    elif s[1] == "radio_emm_from_c":
        try:
            yl = s[2]
            if "T" in yl or "t" in yl:
                sim.radio_emm_from_c = True
            else:
                sim.radio_emm_from_c = False
        except IndexError:
            tools.fatal_error("radio_emm_from_c requires true or false as first argument")
        if sim.radio_emm_from_c:
            try:
                yl = s[3]
                if not isfile(yl):
                    tools.fatal_error("Cannot find the c executable: "+yl+"\nradio_emm_from_c requires true or false as first argument and the path to the c executable as the second")
                sim.exec_emm_c = yl
            except IndexError:
                tools.fatal_error("radio_emm_from_c requires true or false as first argument and the path to the c executable as the second")
    else:
        try:
            if commands[s[1]][1] == "float":
                setattr(commands[s[1]][0],commands[s[1]][2],float(s[2])) 
            elif commands[s[1]][1] == "int":
                setattr(commands[s[1]][0],commands[s[1]][2],int(s[2])) 
            elif commands[s[1]][1].startswith("intarray"):
                setattr(commands[s[1]][0],commands[s[1]][2],array(s[2:],dtype=int)) 
            elif commands[s[1]][1].startswith("floatarray"):
                setattr(commands[s[1]][0],commands[s[1]][2],array(s[2:],dtype=float)) 
            elif commands[s[1]][1].startswith("stringarray"):
                setattr(commands[s[1]][0],commands[s[1]][2],s[2:])
            else: 
                setattr(commands[s[1]][0],commands[s[1]][2],s[2]) 
        except IndexError:
            tools.fatal_error(commands[s[1]][3])
        except ValueError:
            tools.fatal_error(s[1]+" requires a"+commands[s[1]][1]+" value "+commands[s[1]][3])
    

def enumerate_set_commands(commands):
    print("Enumerating all possible set commands")
    for c in commands:
        if c not in ["electrons_from_c","radio_emm_from_c"]:
            print("set "+c+" "+commands[c][1]+" "+commands[c][3])
    print("set electrons_from_c boolean file_path (electrons_from_c requires true or false as first argument and the path to the c executable as the second)")
    print("set radio_emm_from_c boolean file_path (radio_emm_from_c requires true or false as first argument and the path to the c executable as the second)")
        
                

def get_file_id(sim,phys,cos_env,halo,nameFirst,noBfield=False,noGas=False):
    """
    Builds an output file id code
        ---------------------------
        Parameters
        ---------------------------
        sim       - Required : simulation environment (simulation_env)
        phys      - Required : physical environment (phys_env)
        cos_env   - Required : cosmology environment (cosmology_env)
        halo      - Required : halo environment(halo_env)
        nameFirst - Required : a flag that sets halo.name as the first element of the file id
        noBfield  - Optional : if True leave out the B field model details
        noGas     - Optional : if True leave out the gas model details
        ---------------------------
        Output
        ---------------------------
        Unique file ID excluding extension (string)
    """
    if halo.profile == "nfw":
        dm_str = "nfw"
    elif halo.profile == "einasto":
        dm_str = "ein"+str(halo.alpha)
    elif halo.profile == "burkert":
        dm_str = "burkert"
    elif halo.profile == "isothermal":
        dm_str = "isothermal"
    elif halo.profile == "moore":
        dm_str = "moore"
    else:
        dm_str = "gnfw"+str(halo.dm)
    dm_str += "_"

    if(sim.sub_mode == "sc2006"):
        sub_str = "sc2006_fs"+str(halo.sub_frac) #sub halo mass fraction to append to file names
    elif(sim.sub_mode == "prada"):
        sub_str = "prada"
    else:
        sub_str = "none"
    sub_str += "_"

    z_str = "z"+str(halo.z)+"_"  #redshift value to append to file names

    if not noBfield:
        if(phys.diff == 0):
            diff_str = "d0_"
        else:
            diff_str = "d1_"
    else:
        diff_str = ""
    
    wimp_str = phys.particle_model+"_mx"+str(int(phys.mx))
    if halo.mode_exp == 1.0:
        wimp_str += "_"+"decay"
    wimp_str += "_"
    
    if halo.name != None:
        halo_str = halo.name
    else:
        halo_str = "m"+str(int(log10(halo.mvir)))
    halo_str += "_"

    if not noBfield:
        if phys.btag is None:
            if(phys.b0 >= 1.0):
                field_str = "b"+str(int(phys.b0))+"q"+str(phys.qb)
            else:
                field_str = "b"+str(phys.b0)+"q"+str(phys.qb)
        else:
            field_str = phys.btag
        field_str += "_"
    else:
        field_str = ""

    if nameFirst:
        if halo.ucmh == 0:
            file_id = halo_str+wimp_str+z_str+field_str+diff_str+dm_str+sub_str[:-1]
        else:
            file_id = halo_str+wimp_str+z_str+field_str+diff_str[:-1]
    else:
        if halo.ucmh == 0:
            file_id = wimp_str+halo_str+z_str+field_str+diff_str+dm_str+sub_str[:-1]
        else:
            file_id = wimp_str+halo_str+"dL"+str(halo.dl)+"_"+field_str+diff_str[:-1]
    return file_id

def short_id(sim,phys,halo):
    """
    Short file ID using just particle model details
        ---------------------------
        Parameters
        ---------------------------
        sim       - Required : simulation environment (simulation_env)
        phys      - Required : physical environment (phys_env)
        halo      - Required : halo environment(halo_env)
        ---------------------------
        Output
        ---------------------------
        Returns a string of the form 'bb_mx300' for a particle physics model 'bb' and 300 GeV WIMP mass
    """
    wimp_str = phys.particle_model+"_mx"+str(int(phys.mx))
    return wimp_str

#===========================================================================================
def get_emm_nu(halo,phys,sim):
    """
    Calculate and store multi-frequency emissivities at a single frequency
        ---------------------------
        Parameters
        ---------------------------
        halo      - Required : halo environment(halo_env)
        phys      - Required : physical environment (phys_env)
        sim       - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        None - assigned to halo.he_emm_nu
    """ 
    print("=========================================================")
    print("Calculating Emissivity Functions")
    print("=========================================================")
    freq = sim.f_sample
    num = sim.num
    nu = array([sim.nu_sb])
    sim.num = 1
    fnu = [1.4e3,1e12,1e17]
    sim.f_sample = array([fnu[0]])
    halo.radio_emm_nu = radio.radio_emm(halo,phys,sim)[0]
    print("Radio Emissivity at "+str(fnu[0]*1e-3)+" GHz Complete")
    sim.f_sample = array([fnu[1]])
    halo.he_emm_nu = high_e.high_E_emm(halo,phys,sim)[0]
    print("X-ray Emissivity at "+str(fnu[1]*1e-3)+" GHz Complete")
    sim.f_sample = array([fnu[2]])
    halo.gamma_emm_nu = high_e.gamma_source(halo,phys,sim)[0]
    print("Gamma-ray Emissivity at "+str(fnu[2]*1e-3)+" GHz Complete")
    sim.f_sample = freq
    sim.num = num

def get_gamma_emm_nu(halo,phys,sim):
    """
    Calculate and store gamma-ray emissivity at single frequency
        ---------------------------
        Parameters
        ---------------------------
        halo      - Required : halo environment(halo_env)
        phys      - Required : physical environment (phys_env)
        sim       - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        None - assigned to halo.gamma_emm_nu
    """ 
    print("=========================================================")
    print("Calculating Emissivity Functions")
    print("=========================================================")
    freq = sim.f_sample
    num = sim.num
    nu = array([sim.nu_sb])
    sim.num = 1
    sim.f_sample = nu
    halo.gamma_emm_nu = high_e.gamma_source(halo,phys,sim)[0]
    print("Gamma-ray Emissivity at "+str(sim.nu_sb*1e-3)+" GHz Complete")
    sim.f_sample = freq
    sim.num = num

def get_emm(halo,phys,sim):
    """
    Calculate and store multi-frequency emissivity at all frequencies
        ---------------------------
        Parameters
        ---------------------------
        halo      - Required : halo environment(halo_env)
        phys      - Required : physical environment (phys_env)
        sim       - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        None - assigned to halo.radio_emm
    """ 
    if not sim.radio_emm_from_c:
        print("=========================================================")
        print("Calculating Radio Emissivity Functions with Python")
        print("=========================================================")
        halo.radio_emm = radio.radio_emm(halo,phys,sim)
    else:
        print("=========================================================")
        print("Calculating Radio Emissivity Functions with C")
        print("=========================================================")
        py_file = "temp_radio_emm_py.out"
        c_file = "temp_radio_emm_c.in"
        wd = os.getcwd()
        halo.radio_emm = radio.emm_from_c(join(wd,py_file),join(wd,c_file),halo,phys,sim)
        os.remove(join(wd,py_file))
        os.remove(join(wd,c_file))
        if halo.radio_emm is None:
            tools.fatal_error("The radio_emm executable is not compiled/has no specified location")
    print("Process Complete")

def get_nu_emm(halo,phys,sim):
    """
    Calculate and store neutrino emissivity at all frequencies
        ---------------------------
        Parameters
        ---------------------------
        halo      - Required : halo environment(halo_env)
        phys      - Required : physical environment (phys_env)
        sim       - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        None - assigned to halo.nu_emm
    """ 
    print("=========================================================")
    print("Calculating Neutrino Emissivity Functions")
    print("=========================================================")
    halo.nu_emm = neutrino.nu_emm(halo,phys,sim)
    print("Process Complete")

def get_h_emm(halo,phys,sim):
    """
    Calculate and store high-frequency (X-ray and up) emissivity at all frequencies
        ---------------------------
        Parameters
        ---------------------------
        halo      - Required : halo environment(halo_env)
        phys      - Required : physical environment (phys_env)
        sim       - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        None - assigned to halo.he_emm
    """ 
    print("=========================================================")
    print("Calculating High-Energy Emissivity Functions")
    print("=========================================================")
    halo.he_emm = high_e.high_E_emm(halo,phys,sim)
    print("Process Complete")

def get_flux(halo,sim,theta_flag):
    """
    Calculate and store multi-frequency flux at all frequencies
        ---------------------------
        Parameters
        ---------------------------
        halo       - Required : halo environment(halo_env)
        sim        - Required : simulation environment (simulation_env)
        theta_flag - Required : integration region flag 
        ---------------------------
        Output
        ---------------------------
        None - assigned to halo.radio_arcflux or halo.radio_arcflux
    """ 
    print("=========================================================")
    print("Calculating Synchrotron Flux")
    print("=========================================================")
    if theta_flag == "theta" or theta_flag == "angular" or theta_flag == "ang":
        print("Finding Flux Within angular radius of: "+str(sim.theta)+" arcmin")
        halo.radio_arcflux = radio.radio_flux(halo.da*tan(sim.theta*2.90888e-4),halo,sim)
    elif theta_flag == "r_integrate" and not sim.rintegrate is None:
        print("Finding Flux Within radius of: "+str(sim.rintegrate)+" Mpc")
        halo.radio_arcflux = radio.radio_flux(sim.rintegrate,halo,sim)
    else:
        print("Finding Flux within Virial Radius")
        halo.radio_virflux = radio.radio_flux(halo.rvir,halo,sim)
    print('Magnetic Field Average Strength: '+str(halo.bav)+" micro Gauss")
    print('Gas Average Density: '+str(halo.neav)+' cm^-3')
    print("Process Complete")

def get_h_flux(halo,sim,gamma_only,theta_flag,suppress_output=False):
    """
    Calculate and store high-energy mechanism flux at all frequencies
        ---------------------------
        Parameters
        ---------------------------
        halo            - Required : halo environment(halo_env)
        sim             - Required : simulation environment (simulation_env)
        gamma_only      - Required : flag to choose all high-energy or just gamma-ray
        theta_flag      - Required : integration region flag 
        suppress_output - Required : hide output lines
        ---------------------------
        Output
        ---------------------------
        None - assigned to halo.he_arcflux or halo.he_virflux
    """ 
    if not suppress_output:
        print("=========================================================")
        print("Calculating IC and Bremsstrahlung Flux")
        print("=========================================================")
    high_e.gamma_source(halo,phys,sim)
    if theta_flag == "theta" or theta_flag == "angular" or theta_flag == "ang":
        if not suppress_output:
            print("Finding Flux Within angular radius of: "+str(sim.theta)+" arcmin")
        halo.he_arcflux = high_e.high_E_flux(halo.da*tan(sim.theta*2.90888e-4),halo,sim,gamma_only)
    elif theta_flag == "r_integrate" and not sim.rintegrate is None:
        if not suppress_output:
            print("Finding Flux Within radius of: "+str(sim.rintegrate)+" Mpc")
        halo.he_arcflux = high_e.high_E_flux(sim.rintegrate,halo,sim,gamma_only)
    else:
        if not suppress_output:
            print("Finding Flux within Virial Radius")
        halo.he_virflux = high_e.high_E_flux(halo.rvir,halo,sim,gamma_only)
    if not suppress_output:
        print('Magnetic Field Average Strength: '+str(halo.bav)+" micro Gauss")
        print('Gas Average Density: '+str(halo.neav)+' cm^-3')
        print("Process Complete")

def get_nu_flux(halo,sim,gamma_only,theta_flag):
    """
    Calculate and store neutrino flux at all frequencies
        ---------------------------
        Parameters
        ---------------------------
        halo            - Required : halo environment(halo_env)
        sim             - Required : simulation environment (simulation_env)
        gamma_only      - Required : flag to choose all high-energy or just gamma-ray (int)
        theta_flag      - Required : integration region flag (string)
        ---------------------------
        Output
        ---------------------------
        None - halo.nu_arcflux or halo.nu_virflux assigned instead
    """ 
    print("=========================================================")
    print("Calculating Neutrino Flux")
    print("=========================================================")
    if theta_flag == "theta" or theta_flag == "angular" or theta_flag == "ang":
        print("Finding Flux Within angular radius of: "+str(sim.theta)+" arcmin")
        halo.nu_arcflux = neutrino.nu_flux(halo.da*tan(sim.theta*2.90888e-4),halo,sim,gamma_only)
    elif theta_flag == "r_integrate" and not sim.rintegrate is None:
        print("Finding Flux Within radius of: "+str(sim.rintegrate)+" Mpc")
        halo.nu_arcflux = neutrino.nu_flux(sim.rintegrate,halo,sim,gamma_only)
    else:
        print("Finding Flux within Virial Radius")
        halo.nu_virflux = neutrino.nu_flux(halo.rvir,halo,sim,gamma_only)
    print("Process Complete")

def get_multi_flux(halo):
    """
    Compose multi-frequency flux by adding up all mechanisms
        ---------------------------
        Parameters
        ---------------------------
        halo            - Required : halo environment(halo_env)
        ---------------------------
        Output
        ---------------------------
        None - operations performed on halo
    """ 
    halo.make_spectrum()

def get_sync_sb(halo,sim):
    """
    Calculate and store synchrotron surface brightness
        ---------------------------
        Parameters
        ---------------------------
        halo            - Required : halo environment(halo_env)
        sim             - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        Synchrotron surface brightness as a function of angular radius, array-like (float)
    """ 
    print("=========================================================")
    print("Calculating Synchrotron Surface Brightness")
    print("=========================================================")
    print('Magnetic Field Average Strength: '+str(halo.bav)+" micro Gauss")
    print('Gas Average Density: '+str(halo.neav)+' cm^-3')
    return radio.radio_sb(halo,sim)

def get_xray_sb(halo,sim):
    """
    Calculate and store X-ray surface brightness
        ---------------------------
        Parameters
        ---------------------------
        halo            - Required : halo environment(halo_env)
        sim             - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        X-ray surface brightness as a function of angular radius, array-like (float)
    """ 
    print("=========================================================")
    print("Calculating X-ray Surface Brightness")
    print("=========================================================")
    print('Magnetic Field Average Strength: '+str(halo.bav)+" micro Gauss")
    print('Gas Average Density: '+str(halo.neav)+' cm^-3')
    return high_e.xray_sb(halo,sim)

def get_gamma_sb(halo,sim):
    """
    Calculate and store gamma-ray surface brightness
        ---------------------------
        Parameters
        ---------------------------
        halo            - Required : halo environment(halo_env)
        sim             - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        Gamma-ray surface brightness as a function of angular radius, array-like (float)
    """ 
    print("=========================================================")
    print("Calculating Gamma-ray Surface Brightness")
    print("=========================================================")
    return high_e.gamma_sb(halo,sim)

def get_electrons(halo,phys,sim):
    """
    Calculate electron equilibrium distributions
        ---------------------------
        Parameters
        ---------------------------
        halo            - Required : halo environment (halo_env)
        phys            - Required : physical environment (phys_env)
        sim             - Required : simulation environment (simulation_env)
        ---------------------------
        Output
        ---------------------------
        None - assigned to halo.electrons
    """ 
    if not sim.electrons_from_c:
        print("=========================================================")
        print("Calculating Electron Equilibriumn Distributions with Python")
        print("=========================================================")
        print('Magnetic Field Average Strength: '+str(halo.bav)+" micro Gauss")
        print('Gas Average Density: '+str(halo.neav)+' cm^-3')
        halo.electrons = electron.equilibrium_p(halo,phys)
    else:
        print("=========================================================")
        print("Calculating Electron Equilibriumn Distributions with C")
        print("=========================================================")
        print('Magnetic Field Average Strength: '+str(halo.bav)+" micro Gauss")
        print('Gas Average Density: '+str(halo.neav)+' cm^-3')
        py_file = "temp_electrons_py.out"
        c_file = "temp_electrons_c.in"
        wd = os.getcwd()
        halo.electrons = electron.electrons_from_c(join(wd,py_file),join(wd,c_file),halo,phys,sim)
        #os.remove(join(wd,py_file))
        #os.remove(join(wd,c_file))
        if halo.electrons is None:
            tools.fatal_error("The electron executable is not compiled/location not specified")
    print("Process Complete")

def loop_mass(argset):
    """
    This calculates the emissions from a halo - used for pooling tasks in a loop over halos
        ---------------------------
        Parameters
        ---------------------------
        argset - Required : [halo,sim,phys]
        ---------------------------
        Output
        ---------------------------
        None - assigned to halo variables
    """
    halo = argset[0]; sim = argset[1]; phys = argset[2]
    get_electrons(halo,phys,sim)
    get_emm(halo,phys,sim)
    get_flux(halo,sim,"full")


def read_spectrum(spec_file,gamma,branching,phys,sim):
    """
    This extracts dN/dE spectra from a specified file, gamma flags what kind of spectrum it is 
        ---------------------------
        Parameters
        ---------------------------
        spec_file - Required : the path to the data file
        gamma     - Required : flag to say what kind of spectrum 0 -> positrons, 1 -> gamma-ray, 2 -> neutrino
        branching - Required : branching ratio to be applied to this file's data
        phys      - Required : physical environment (phys_env) 
        sim       - Required : simulation environment (simulation_env) 
        ---------------------------
        Outputs
        ---------------------------
        None - assigned to spectrum variables of phys
    """
    bin_flag = True
    if sim.e_bins is None:
        bin_flag = False
    else:
        spec_length = sim.e_bins
    try:
        spec = open(spec_file,'r')
    except IOError:
        tools.fatal_error("Spectrum File: "+spec_file+" does not exist at the specified location")
    me = 0.511e-3
    s = []
    for line in spec:
        if not line.startswith("#"):
            useLine = line.strip().split("#")[0]
            s.append(useLine.strip().split())
    E_set = zeros(len(s),dtype=float)  #Energies set
    Q_set = zeros(len(s),dtype=float)  #electron generation function dn/dE
    for i in range(0,len(s)):
        #me factors convert this to gamma and dn/dgamma
        #gamma is the Lorenz factor for electrons with energy E
        E_set[i] = float(s[i][0])/me
        Q_set[i] = float(s[i][1])*me
    spec.close()
    if not bin_flag:
        sim.e_bins = len(E_set)
        spec_length = sim.e_bins
        
    if(gamma == 0):
        if phys.spectrum[0] is None and phys.spectrum[1] is None:
            phys.specMin = E_set[0]
            phys.specMax = E_set[-1]
            phys.spectrum[0] = logspace(log10(phys.specMin*1.00001),log10(phys.specMax*0.99999),num=spec_length)
            phys.spectrum[1] = zeros(spec_length)
        intSpec = interp1d(E_set,Q_set)
        newE = phys.spectrum[0]
        Q_set = intSpec(newE)
        phys.spectrum[1] += Q_set*branching
    elif gamma == 1:
        if phys.gamma_spectrum[0] is None and phys.gamma_spectrum[1] is None:
            phys.gamma_specMin = E_set[0]
            phys.gamma_specMax = E_set[-1]
            phys.gamma_spectrum[0] = logspace(log10(phys.gamma_specMin*1.00001),log10(phys.gamma_specMax*0.99999),num=spec_length)
            phys.gamma_spectrum[1] = zeros(spec_length)
        intSpec = interp1d(E_set,Q_set)
        newE = phys.gamma_spectrum[0]
        Q_set = intSpec(newE)
        phys.gamma_spectrum[1] += Q_set*branching
    elif gamma == 2:
        phys.nu_spectrum[0] = E_set
        phys.nu_spectrum[1] = Q_set

def gather_spectrum(spec_dir,phys,sim,mode="ann"):
    """
    This extracts dN/dE spectra from input data in environments by working out what file to use
        ---------------------------
        Parameters
        ---------------------------
        spec_dir - Required : the path to the data file storage directory
        phys     - Required : physical environment (phys_env) 
        sim      - Required : simulation environment (simulation_env) 
        mode     - Required : annihilation or decay
        ---------------------------
        Output
        ---------------------------
        None - executes read_spectrum
    """
    if mode == "ann":
        for ch,br in zip(phys.channel,phys.branching):
            pos = join(spec_dir,"pos_"+ch+"_"+str(int(phys.mx))+"GeV.data")
            gamma = join(spec_dir,"gamma_"+ch+"_"+str(int(phys.mx))+"GeV.data")
            read_spectrum(pos,0,br,phys,sim)
            read_spectrum(gamma,1,br,phys,sim)
    else:
        for ch,br in zip(phys.channel,phys.branching):
            pos = join(spec_dir,"pos_"+ch+"_"+str(int(phys.mx*0.5))+"GeV.data")
            gamma = join(spec_dir,"gamma_"+ch+"_"+str(int(phys.mx*0.5))+"GeV.data")
            read_spectrum(pos,0,br,phys,sim)
            read_spectrum(gamma,1,br,phys,sim)

def neutrino_spectrum(spec_dir,phys,sim,flavour,mode="ann"):
    """
    This extracts dN/dE neutrino spectra from input data in environments by working out what file(s) to use
        ---------------------------
        Parameters
        ---------------------------
        spec_dir - Required : the path to the data file storage directory
        phys     - Required : physical environment (phys_env) 
        sim      - Required : simulation environment (simulation_env) 
        flavour  - Required : neutrino flavour to use
        mode     - Required : annihilation or decay
        ---------------------------
        Output
        ---------------------------
        None - assigned to phys.nu_spectrum 
    """
    me = 0.511e-3
    for ch,br in zip(phys.channel,phys.branching):
        if mode == "ann":
            nu_e = join(spec_dir,"nu_e_"+ch+"_"+str(int(phys.mx))+"GeV.data")
            nu_mu = join(spec_dir,"nu_mu_"+ch+"_"+str(int(phys.mx))+"GeV.data")
            nu_tau = join(spec_dir,"nu_tau_"+ch+"_"+str(int(phys.mx))+"GeV.data")
        else:
            nu_e = join(spec_dir,"nu_e_"+ch+"_"+str(int(phys.mx*0.5))+"GeV.data")
            nu_mu = join(spec_dir,"nu_mu_"+ch+"_"+str(int(phys.mx*0.5))+"GeV.data")
            nu_tau = join(spec_dir,"nu_tau_"+ch+"_"+str(int(phys.mx*0.5))+"GeV.data")
        if flavour == "mu":
            nu_set = [nu_mu]
        elif flavour == "e":
            nu_set = [nu_e]
        elif flavour == "tau":
            nu_set = [nu_tau]
        else:
            nu_set = [nu_e,nu_mu,nu_tau]
        for nu in nu_set:
            try:
                spec = open(nu,"r")
            except IOError:
                tools.fatal_error("Input file: "+nu+" could not be found")
            s = []
            for line in spec:
                if not line.startswith("#"):
                    s.append(line.strip().split())
            E_set = zeros(len(s),dtype=float)  #Energies set
            Q_set = zeros(len(s),dtype=float)  #electron generation function dn/dE
            for i in range(0,len(s)):
                #me factors convert this to gamma and dn/dgamma
                #gamma is the Lorenz factor for an electron
                E_set[i] = float(s[i][0])/me
                Q_set[i] = float(s[i][1])*me
            spec.close()
            if phys.nu_spectrum[0] is None and phys.nu_spectrum[1] is None:
                phys.nu_specMin = E_set[0]
                phys.nu_specMax = E_set[-1]
                if sim.e_bins == None:
                    sim.e_bins = len(E_set)
                nu_spec = [None,zeros(sim.e_bins)]
                nu_spec[0] = logspace(log10(phys.nu_specMin*1.00001),log10(phys.nu_specMax*0.99999),num=sim.e_bins)
                phys.nu_spectrum[0] = nu_spec[0]
                phys.nu_spectrum[1] = zeros(len(phys.nu_spectrum[0]))
            intSpec = interp1d(E_set,Q_set)
            newE = phys.nu_spectrum[0]
            Q_set = intSpec(newE)
            phys.nu_spectrum[1] += Q_set*br
    phys.nu_flavour = flavour

def console_mode(phys,sim,halo,cosmo,loop):
    print("Dark matters is now in console mode")
    print("Enter a command: ")
    command_line = ""
    while not command_line.strip().lower == "exit": 
        command_line = input("> ")
        process_command(command_line,phys,sim,halo,cosmo,loop)


try:
    args = sys.argv[1:]
    in_file = args[0]
    test = open(in_file,"r")
    console_flag = False
    test.close()
except IndexError:
    console_flag = True
except IOError:
    tools.fatal_error("Invalid script file path supplied")


#===============================================================
#Code execution
#===============================================================
phys = physical_env() #set up default physical environment
cosm = cosmology_env() #set up default cosmology environment
sim = simulation_env() #set up default simulation environment
loop = loop_env(zn=20,zmin=1.0e-5,zmax=1.0,nloop=40,mmin=1.1e-6,mmax=1e15) #loop env - old code
loop_sim = simulation_env() #sets the common values for the loop
halo = halo_env()  #set up default halo environment
if not console_flag:
    process_file(in_file,phys,sim,halo,cosm,loop) #execute all commands in the input file
else:
    console_mode(phys,sim,halo,cosm,loop)