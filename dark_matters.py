import sys,getopt,time,os
from copy import deepcopy
try:
    import numpy as np
except:
    print("Fatal Error: the numpy package is missing")
    sys.exit(2)
try:
    from scipy.interpolate import interp1d,interp2d
except:
    print("Fatal Error: the scipy package is missing")
    sys.exit(2)
try:
    import astropy
except:
    print("Fatal Error: the astropy package is missing")
    sys.exit(2)
try:
    from wimp_tools import tools,cosmology_env,simulation_env,physical_env,loop_env,halo_env #wimp handling package
except:
    print("Fatal Error: the wimp_tools sub-package of dark_matters is missing")
    sys.exit(2)
try:
    import output #output module
except:
    print("Fatal Error: file output.py is missing!")
    sys.exit(2)
try:
    import calc_manager #output module
except:
    print("Fatal Error: file calc_manager.py is missing!")
    sys.exit(2)
from multiprocessing import Pool
from os.path import join,isfile,isdir

from astropy.io import fits
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
calculations = []
last_set = []
spacerStr = "========================================================="
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
    global calculations
    global last_set
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
    global calculations
    global last_set,spacerStr
    calc_dict = {"dflux":"Gamma-ray flux from D-factor","rflux":"Radio flux","hflux":"High-frequency flux","jflux":"Gamma-ray flux from Jfactor","gflux":"Gamma-ray flux","nuflux":"Neutrino flux","nu_jflux":"Neutrino flux from J-factor","sb":"Radio surface brightness","gamma_sb":"gamma-ray surface brightness","emm":"Radio Emmissivity","electrons":"Electron distributions","flux":"All fluxes"}
    diff_calc_set = ["rflux","hflux","flux","sb","emm","electrons","flux"]
    sb_dict = {"sb":"rflux"}
    if "#" in command.strip() and not command.strip().startswith("#"):
        s = command.split("#")[0].split()
    elif not command.strip() == "":
        s = command.split()
    if command.strip().startswith("#") or command.strip() == "":
        pass
    elif command.strip() in ["clear calc","clear c","clear calculation","clear calculations"]:
        calculations = []
    elif command.strip() in ["display calc", "display c","show c","show calc"]:
        if not calculations == []:
            print(spacerStr)
            print("Displaying calculation set:")
            print(spacerStr)
            for i in range(len(calculations)):
                print("Index:{} Label:{}".format(i,calculations[i].calcLabel))
        else:
            print("No calculations have been produced yet!")
    elif command.strip().lower() == "calc help":
        print("The calc command is used as follows: calc mode region")
        print("region -> full, theta, r_integrate")
        print("mode can take the values")
        for c in calc_dict:
            print(c+" calculates "+calc_dict[c])
    elif s[0].lower() in ["out","o","output"]:
        print(spacerStr)
        print("Preparing to output flux data")
        print(spacerStr)
        if calculations == []:
            tools.fatal_error("no calculations have been produced to output yet!")
        try:
            if not s[1].lower() in calc_dict:
                tools.fatal_error("command "+s[0].lower()+" must be supplied with a fluxmode from: "+calc_dict.keys())
            else:
                calc_mode = s[1].lower()
        except:
            tools.fatal_error("command "+s[0].lower()+" must be supplied with a fluxmode from: "+calc_dict.keys())
        try:
            theta_flag = s[2].lower()
        except:
            if not "jflux" in calc_mode: 
                print("Warning: no region flag supplied with command \'"+s[0].lower()+"\' defaulting to \'full\'")
            theta_flag = "full"
        indices = getIndex(s) 
        if not "jflux" in calc_mode:
            print("Region: "+theta_flag)
        else:
            print("Region: from Jfactor")
        print("Flux command: "+calc_dict[s[1]])
        full_id = True
        if "jflux" in calc_mode or "gflux" in calc_mode:
            full_id = False
        for i in indices:
            if calc_mode in diff_calc_set:
                rmax = getRmax(theta_flag,calculations[i],command)
                calculations[i].halo.physical_averages(rmax)
            print("Calculation ID: "+output.getCalcID(calculations[i].sim,calculations[i].phys,calculations[i].cosmo,calculations[i].halo,short_id=(not full_id)))
            if theta_flag == "radial":
                calculations[i].calcFluxRadial(sim.nu_sb,calc_mode,full_id=full_id,suppress_output=False)
            elif theta_flag == "sb":
                calculations[i].calcSB(sim.nu_sb,calc_mode,full_id=full_id,suppress_output=False)
            else:
                calculations[i].calcFlux(calc_mode,regionFlag=theta_flag,full_id=full_id,suppress_output=False)
    elif s[0].lower() in ["fromfits","ff"]:
        #emm: add to sim,phys,halo,cosmo from the hdul[0].header then loop over calculations with elements of hdul[0].data, finally store as fits file
        try:
            hdul = fits.open(s[1])
        except IOError:
            tools.fatal_error("Fits file: {} was not found".format(s[1]))
        hdr = hdul[0].header
        hdd = hdul[0].data
        halo.haloFromHeader(hdr)
        phys.physFromHeader(hdr)
        sim.simFromHeader(hdr)
        cos_env.cosmoFromHeader(hdr)
        mxSet = np.array(hdr['CRSET1'].split(),dtype=np.float64)
        fluxMode = hdr['DMCALC'].split("_")[0]
        readCalcSet = []
        fluxCalcSet = []
        try:
            regionFlag = s[2]
        except:
            regionFlag = "full"
        for i in range(len(hdd)):
            phys.mx = mxSet[i]
            readCalcSet.append(calc_manager.calculation(deepcopy(halo),deepcopy(phys),deepcopy(sim),deepcopy(cos_env)))
        for c_run in readCalcSet:
            c_run.calcFluxFromFits(hdd[i],fluxMode,regionFlag)
            fluxCalcSet.append(deepcopy(c_run))
        output.fitsFlux(fluxCalcSet,fluxMode,regionFlag,header=hdr)
    elif s[0].lower() in ["f","fits"]:
        print(spacerStr)
        print("Preparing to output fits data cubes")
        print(spacerStr)
        fitsModes = ["electrons","emmisivity","emm","flux","sb"]
        if calculations == []:
            tools.fatal_error("no calculations have been produced to output yet!")
        try:
            if not s[1].lower() in fitsModes:
                tools.fatal_error(("command "+s[0].lower()+" must be supplied with a an output type from: ").join(str(f) for f in fitsModes))
            else:
                calc_mode = s[1].lower()
        except:
            tools.fatal_error(("command "+s[0].lower()+" must be supplied with an output type from: ").join(str(f) for f in fitsModes))
        try:
            fitsName = s[3].lower()
            try:
                int(fitsName)
                startIndex = 3
                fitsName = None
            except:
                startIndex = 4
        except:
            fitsName = None
            startIndex = 3
        indices = getIndex(s,startIndex=startIndex)
        print("Output command: "+s[1])
        full_id = True
        runSet = []
        if calc_mode == "electrons":
            for i in indices:
                if calculations[i].halo.electrons is None:
                    tools.fatal_error("All runs must have had electron distributions calculated")
                else:
                    runSet.append(deepcopy(calculations[i]))
            output.fitsElectron(runSet)
        elif calc_mode == "sb":
            try:
                if not s[2].lower() in calc_dict:
                    tools.fatal_error("command "+s[1].lower()+" must be supplied with a fluxmode from: "+calc_dict.keys())
                else:
                    calc_mode = s[1].lower()
            except:
                tools.fatal_error("command "+s[1].lower()+" must be supplied with a fluxmode from: "+calc_dict.keys())
            for i in indices:
                emmDict = {"rflux":"radio_emm","gflux":"gamma_emm","hflux":"he_emm","nuflux":"nu_emm"}
                if getattr(calculations[i].halo,emmDict[s[2].lower()]) is None:
                    tools.fatal_error("All runs must have had {} emmisivity computed".format(s[2].lower()))
                else:
                    runSet.append(deepcopy(calculations[i]))
            output.fitsSB(runSet,s[2].lower(),fName=fitsName)
        elif calc_mode == "flux":
            try:
                if not s[2].lower() in calc_dict:
                    tools.fatal_error("command "+s[1].lower()+" must be supplied with a fluxmode from: "+calc_dict.keys())
                else:
                    calc_mode = s[1].lower()
            except:
                tools.fatal_error("command "+s[1].lower()+" must be supplied with a fluxmode from: "+calc_dict.keys)
            for i in indices:
                emmDict = {"rflux":"radio_emm","gflux":"gamma_emm","hflux":"he_emm","nuflux":"nu_emm"}
                if getattr(calculations[i].halo,emmDict[s[2].lower()]) is None:
                    tools.fatal_error("All runs must have had {} emmisivity computed".format(s[2].lower()))
                else:
                    runSet.append(deepcopy(calculations[i]))
            output.fitsSB(runSet,s[2].lower(),fName=fitsName)
        else:
            try:
                if not s[2].lower() in calc_dict:
                    tools.fatal_error("command "+s[1].lower()+" must be supplied with a fluxmode from: "+calc_dict.keys())
                else:
                    calc_mode = s[1].lower()
            except:
                tools.fatal_error("command "+s[1].lower()+" must be supplied with a fluxmode from: "+calc_dict.keys())
            for i in indices:
                emmDict = {"rflux":"radio_emm","gflux":"gamma_emm","hflux":"he_emm","nuflux":"nu_emm"}
                if getattr(calculations[i].halo,emmDict[s[2].lower()]) is None:
                    tools.fatal_error("All runs must have had {} emmisivity computed".format(s[2].lower()))
                else:
                    runSet.append(deepcopy(calculations[i]))
            output.fitsEmm(runSet,s[2].lower())
        del(runSet)
    elif s[0].lower() in ["c+o","calc+out","calc+output"]:
        try:
            print("Calculation Task Given: "+calc_dict[s[1]])
        except IndexError:
            tools.fatal_error("calc+out must be supplied with an option: "+" ".join(key for key in calc_dict.keys()))
        except KeyError:
            tools.fatal_error("calc+out must be supplied with a valid option from: "+" ".join(key for key in calc_dict.keys()))
        try:
            calc_str = s[1]
            try:
                t_str = s[2]
            except:
                t_str = "full"
        except:
            calc_str = "flux"
            t_str = "full"
        last_set = []
        batch_calc = 0
        batch_time = time.time()
        print(spacerStr)
        print("Initialising calculations data")
        print(spacerStr)
        for m in sim.mx_set:
            phys.clear_spectra()
            batch_calc += 1
            process_command("set wimp_mass "+str(m),phys,sim,halo,cos_env,loop)
            if sim.specdir is None:
                tools.fatal_error("Please specify a directory for input spectra with 'set input_spectra_directory #' command")
            if(not halo.check_self()):
                tools.fatal_error("halo data underspecified")
            if(not phys.check_self()):
                tools.fatal_error("physical environment data underspecified")
            if(not sim.check_self()):
                tools.fatal_error("simulation data underspecified")
            if "nu" in calc_str:
                use_nu = True
            else:
                use_nu = False
            #    neutrino_spectrum(sim.specdir,phys,sim,sim.nu_flavour,mode=halo.mode)
            if(not phys.check_particles()):
                tools.fatal_error("particle data underspecified")
            #gather_spectrum(sim.specdir,phys,sim,mode=halo.mode)
            getSpectralData(sim.specdir,phys,sim,mode=halo.mode,use_nu=use_nu)
            #print(phys.gamma_spectrum)
            halo.reset_avgs()
            if(not halo.setup(sim,phys,cos_env)):
                tools.fatal_error("halo setup error")
            last_set.append(calc_manager.calculation(deepcopy(halo),deepcopy(phys),deepcopy(sim),deepcopy(cos_env)))
            #print("c "+calc_str+" "+t_st)
        print("All input data consistency checks complete")
        for c_run in last_set:
            print(spacerStr)
            print("Beginning new Dark Matters calculation")
            print(spacerStr)
            if "sb" in calc_str:
                c_run.calcWrite(fluxMode=sb_dict[calc_str])
            else:
                c_run.calcWrite(fluxMode=calc_str)
            if calc_str in diff_calc_set:
                rmax = getRmax(t_str,c_run,command)
                c_run.halo.physical_averages(rmax)
            full_id = True
            if "jflux" in calc_str or "gflux" in calc_str:
                full_id = False
            if t_str == "radial":
                c_run.calcFluxRadial(sim.nu_sb,calc_str,full_id=full_id,suppress_output=False)
            elif t_str == "sb":
                c_run.calcSB(sim.nu_sb,calc_str,full_id=full_id,suppress_output=False)
            else:
                c_run.calcFlux(calc_str,regionFlag=t_str,full_id=full_id,suppress_output=False)
            calculations.append(deepcopy(c_run))
        print("Batch Time: "+str(time.time()-batch_time)+" s")
        print("Batch Time per Calculation: "+str((time.time()-batch_time)/batch_calc)+" s")
    elif s[0].lower() in ["batch","b","c","calc"]:
        try:
            print("Calculation Task Given: "+calc_dict[s[1]])
        except IndexError:
            tools.fatal_error("calc must be supplied with an option: "+str(calc_dict.keys))
        except KeyError:
            tools.fatal_error("calc must be supplied with a valid option from: "+str(calc_dict.keys))
        try:
            calc_str = s[1]
            try:
                t_str = s[2]
            except:
                t_str = "full"
        except:
            calc_str = "flux"
            t_str = "full"
        last_set = []
        batch_calc = 0
        batch_time = time.time()
        print(spacerStr)
        print("Initialising calculations data")
        print(spacerStr)
        for m in sim.mx_set:
            phys.clear_spectra()
            batch_calc += 1
            process_command("set wimp_mass "+str(m),phys,sim,halo,cos_env,loop)
            if sim.specdir is None:
                tools.fatal_error("Please specify a directory for input spectra with 'set input_spectra_directory #' command")
            if(not halo.check_self()):
                tools.fatal_error("halo data underspecified")
            if(not phys.check_self()):
                tools.fatal_error("physical environment data underspecified")
            if(not sim.check_self()):
                tools.fatal_error("simulation data underspecified")
            if "nu" in calc_str:
                use_nu = True
            else:
                use_nu = False
            #    neutrino_spectrum(sim.specdir,phys,sim,sim.nu_flavour,mode=halo.mode)
            if(not phys.check_particles()):
                tools.fatal_error("particle data underspecified")
            #gather_spectrum(sim.specdir,phys,sim,mode=halo.mode)
            getSpectralData(sim.specdir,phys,sim,mode=halo.mode,use_nu=use_nu)
            #print(phys.gamma_spectrum)
            halo.reset_avgs()
            if(not halo.setup(sim,phys,cos_env)):
                tools.fatal_error("halo setup error")
            last_set.append(calc_manager.calculation(deepcopy(halo),deepcopy(phys),deepcopy(sim),deepcopy(cos_env)))
            #print("c "+calc_str+" "+t_st)
        print("All input data consistency checks complete")
        for c_run in last_set:
            print(spacerStr)
            print("Beginning new Dark Matters calculation")
            print(spacerStr)
            c_run.calcWrite(fluxMode=calc_str)
            if calc_str in diff_calc_set:
                rmax = getRmax(t_str,c_run,command)
                c_run.halo.physical_averages(rmax)
            c_run.calcEmm(calc_str)
            calculations.append(deepcopy(c_run))
        print("Batch Time: "+str(time.time()-batch_time)+" s")
        print("Batch Time per Calculation: "+str((time.time()-batch_time)/batch_calc)+" s")
    elif(s[0] == "set"):
        process_set(command,phys,sim,halo,cos_env,loop)
    elif(s[0] == "help" or s[0] == "h"):
        try:
            tools.help_menu()
        except IndexError:
            tools.help_menu()
    elif(s[0] == "load" or s[0] == "l"):
        try:
            open(s[1],"r").close()
            process_file(s[1],phys,sim,halo,cos_env,loop)
        except IndexError:
            tools.fatal_error("the 'load' command requires a valid input file path to be supplied")
    elif(s[0] == "exit" or s[0] == "quit" or s[0] == "q"):
        sys.exit(2)
    elif(s[0] == "show"):
        try:
            if(s[1] == "log"):
                log = output.getCalcID(sim,phys,cos_env,halo,False)+".log"
                tools.write(log,"flux",sim,phys,cos_env,halo)
        except IndexError:
            tools.fatal_error("show must be followed by a valid command")
    else:
        print("Invalid Command: "+command)

def getIndex(s,startIndex=3):
    try:
        indices = s[startIndex:].lower()
    except:
        try:
            if ":" in s[3]:
                imax = int(s[3].split(":")[1])
                imin = int(s[3].split(":")[0])
                if imax == -1:
                    imax = len(calculations)-1
                indices = np.arange(imin,imax+1)
            else:
                indices = [s[3].lower()]
        except:
            print("Warning: no set of calculations to output supplied with command \'"+s[0].lower()+"\' defaulting to \'all\'")
            indices = ["all"]
    if indices == ["all"]:
        indices = np.arange(0,len(calculations))
    elif indices == ["-1"]:
        indices = [len(calculations)-1]
    elif indices == ["last"]:
        imin = len(calculations)-len(last_set)
        imax = len(calculations)-1
        indices = np.arange(imin,imax+1)
    else:
        try:
            indices = np.array(indices,dtype=int)
        except ValueError:
            tools.fatal_error("The indices for output supplied as "+indices+" are not valid")
    return indices

def getRmax(t_str,c_run,command):
    if t_str == "full":
        rmax = c_run.halo.rvir
    elif t_str == "theta":
        rmax = c_run.halo.da*np.tan(c_run.sim.theta*2.90888e-4)
    elif t_str == "r_integrate":
        rmax = c_run.sim.rintegrate
    elif t_str == "radial":
        rmax = c_run.halo.rvir
    elif t_str == "sb":
        if not c_run.sim.theta is None:
            rmax = c_run.halo.da*np.tan(c_run.sim.theta*2.90888e-4)
        elif not c_run.sim.rintegrate is None:
            rmax = c_run.sim.rintegrate
        else:
            rmax = c_run.halo.rvir
    else:
        rmax = c_run.halo.rvir
        print("Warning: region flag "+t_str+" in command "+command+" not recognised, defaulting to \'full\'")
    return rmax

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
    hsp_commands = {"rvir":[halo,"float","rvir","(mpc units)"],"mvir":[halo,"float","mvir","(solar mass units)"],"r_integrate":[sim,"float","rintegrate","(Mpc units)"],"rcore":[halo,"float","rcore","(mpc units)"],"r_stellar_half_light":[halo,"float","r_stellar_half_light","(mpc units)"],"qb":[phys,"float","qb","(function varies by profile)"],"b_model":[phys,"string","b_flag","(chosen magnetic field model)"],"B":[phys,"float","b0","(micro Gauss units)"],"particle_model":[phys,"string","particle_model","(name for particle physics model)"],"ne_model":[phys,"string","ne_model","(chosen electron model)"],"ne":[phys,"float","ne0","(cm^-3 units)"],"qe":[phys,"float","qe","(function varies by ne model)"],"jflag":[halo,"int","J_flag","(jfactor norm flag)"],"btag":[phys,"string","btag","(magnetic field label)"],"theta":[sim,"float","theta","(arcminute units)"],"profile":[halo,"string","profile","(halo density profile)"],"input_spectra_directory":[sim,"string","specdir","(directory with annihilation channel input spectra)"],"alpha":[halo,"float","alpha","(Einasto alpha)"],"f_num":[sim,"int","num","(number of frequency samples)"],"r_num":[sim,"int","n","(number of r samples)"],"gr_num":[sim,"int","ngr","(r samples in Green's functions)"],"e_bins":[sim,"int","e_bins","(number of energy bins for input spectra)"],"submode":[sim,"string","sub_mode","(substructure model)"],"diff":[phys,"int","diff","diffusion flag 1 or 0)"],"name":[halo,"string","name","(halo name)"],"d":[phys,"float","lc","(units kpc)"],"dist":[halo,"float","dl","(units mpc)"],"z":[halo,"float","z","(redshift)"],"nu_flavour":[sim,"string","nu_flavour","(neutrino flavour)"],"channel":[phys,"stringarray","channel","(annihilation channel)"],"branching":[phys,"floatarray","branching","(set of branching ratios in same order as channels, separated by spaces)"],"rhos":[halo,"float","rhos","(unitless)"],"rho0":[halo,"float","rho0","(msol mpc^-3)"],"mx_set":[sim,"floatarray","mx_set","(set of WIMP masses in GeV separated by spaces)"],"flim":[sim,"floatarray","flim","(min and max frequencies in MHz separated by white space)"],"ucmh":[halo,"int","ucmh","(ucmh flag 1 or 0)"],"sub_frac":[halo,"float","sub_frac","(mass fraction of sub-halos)"],"dfactor":[halo,"float","Dfactor","(units of GeV cm^-2)"],"jfactor":[halo,"float","J","(units of GeV^2 cm^-5)"],"cvir":[halo,"float","cvir","(virial concentration)"],"t_star_formation":[halo,"float","t_sf","(star formation time in seconds)"],"b_average":[halo,"float","bav","(micro Gauss)"],"ne_average":[halo,"float","neav","(cm^-3)"],"wimp_mode":[halo,"string","mode","(ann or decay)"],"radio_boost":[sim,"int","radio_boost_flag","(1 or 0)"],"frequency":[sim,"float","nu_sb","(MHz)"],"delta":[phys,"float","delta","(turbulence index)"],"wimp_mass":[phys,"float","mx","(WIMP mass in GeV)"],"output_log":[sim,"int","log_mode","(1 or 0)"],"electrons_from_c":[],"radio_emm_from_c":[],"ne_scale":[phys,"float","lb","(ne scale length Mpc)"]}

    cosmo_commands = {"omega_m":[cos_env,"float","w_m","(matter fraction)"],'omega_lambda':[cos_env,"float","w_l","(cosmological constant fraction)"],'ps_index':[cos_env,"float","n","(matter power spectrum index)"],'curvature':[cos_env,"string","universe","(flat or or curved)"],'omega_b':[cos_env,"float","w_b","(baryon fraction)"],'omega_dm':[cos_env,"float","w_dm","(DM fraction)"],'sigma_8':[cos_env,"float","sigma_8","(power spectrum normalisation)"], 'omega_nu':[cos_env,"float","w_nu","(neutrino fraction)"],'N_nu':[cos_env,"float","N_nu","(neutrino number)"]}

    loop_commands = {"nloop":[loop,"int","mn","(number of mass samples in loop)"]}

    hsp2_commands = {"model_independent":[phys,"boolean","model_independent","(1 or 0 flag for model independent approach)"],"diffusion_constant":[phys,"float","d0","(diffusion constant (cm^2 s^-1))"],"isrf":[phys,"int","ISRF","(inter-sellar radiation field flag (1 or 0))"],"output_directory":[sim,"string","out_dir","(output directory)"],"nfw_index":[halo,"float","gnfw_gamma","(gnfw gamma index)"],"halo_weights":[halo,"string","weights","(Weights for averaging physical quantities)"]}

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
    # elif s[1] == "radio_emm_from_c":
    #     try:
    #         yl = s[2]
    #         if "T" in yl or "t" in yl:
    #             sim.radio_emm_from_c = True
    #         else:
    #             sim.radio_emm_from_c = False
    #     except IndexError:
    #         tools.fatal_error("radio_emm_from_c requires true or false as first argument")
    #     if sim.radio_emm_from_c:
    #         try:
    #             yl = s[3]
    #             if not isfile(yl):
    #                 tools.fatal_error("Cannot find the c executable: "+yl+"\nradio_emm_from_c requires true or false as first argument and the path to the c executable as the second")
    #             sim.exec_emm_c = yl
    #         except IndexError:
    #             tools.fatal_error("radio_emm_from_c requires true or false as first argument and the path to the c executable as the second")
    else:
        try:
            if commands[s[1]][1] == "float":
                setattr(commands[s[1]][0],commands[s[1]][2],float(s[2])) 
            elif commands[s[1]][1] == "int":
                setattr(commands[s[1]][0],commands[s[1]][2],int(s[2])) 
            elif commands[s[1]][1] == "boolean":
                setattr(commands[s[1]][0],commands[s[1]][2],bool(int(s[2]))) 
            elif commands[s[1]][1].startswith("intarray"):
                setattr(commands[s[1]][0],commands[s[1]][2],np.array(s[2:],dtype=int)) 
            elif commands[s[1]][1].startswith("floatarray"):
                setattr(commands[s[1]][0],commands[s[1]][2],np.array(s[2:],dtype=float)) 
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
    #print("set radio_emm_from_c boolean file_path (radio_emm_from_c requires true or false as first argument and the path to the c executable as the second)")

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
    E_set = np.zeros(len(s),dtype=float)  #Energies set
    Q_set = np.zeros(len(s),dtype=float)  #electron generation function dn/dE
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
            phys.spectrum[0] = np.logspace(np.log10(phys.specMin*1.00001),np.log10(phys.specMax*0.99999),num=spec_length)
            phys.spectrum[1] = np.zeros(spec_length)
        intSpec = interp1d(E_set,Q_set)
        newE = phys.spectrum[0]
        Q_set = intSpec(newE)
        phys.spectrum[1] += Q_set*branching
    elif gamma == 1:
        if phys.gamma_spectrum[0] is None and phys.gamma_spectrum[1] is None:
            phys.gamma_specMin = E_set[0]
            phys.gamma_specMax = E_set[-1]
            phys.gamma_spectrum[0] = np.logspace(np.log10(phys.gamma_specMin*1.00001),np.log10(phys.gamma_specMax*0.99999),num=spec_length)
            phys.gamma_spectrum[1] = np.zeros(spec_length)
        intSpec = interp1d(E_set,Q_set)
        newE = phys.gamma_spectrum[0]
        Q_set = intSpec(newE)
        phys.gamma_spectrum[1] += Q_set*branching
    elif gamma == 2:
        phys.nu_spectrum[0] = E_set
        phys.nu_spectrum[1] = Q_set

def getSpectralData(spec_dir,phys,sim,mode="ann",use_nu=False):
    if use_nu:
        specFiles = ["positrons","gammas","neutrinos_{}".format(sim.nu_flavour)]
    else:
        specFiles = ["positrons","gammas"]

    for f in specFiles:
        if phys.model_independent == 1:
            readSpectrum(join(spec_dir,"AtProduction_{}.dat".format(f)),phys,sim,mode=mode)
        else:
            readSpectrum(join(spec_dir,"{}_AtProduction_{}.dat".format(phys.particle_model,f)),phys,sim,mode=mode)

def readSpectrum(spec_file,phys,sim,mode="ann"):
    if phys.model_independent:
        #mDM      Log[10,x]   eL         eR         e          \[Mu]L     \[Mu]R     \[Mu]      \[Tau]L    \[Tau]R    \[Tau]     q            c            b            t            WL          WT          W           ZL          ZT          Z           g            \[Gamma]    h           \[Nu]e     \[Nu]\[Mu]   \[Nu]\[Tau]   V->e       V->\[Mu]   V->\[Tau]
        chCols = {"ee":4,"mumu":7,"tautau":10,"qq":11,"bb":13,"tt":14,"ww":17,"zz":20,"gamma":22,'hh':23}
        mCol = 0
        xCol = 1
        try:
            specData = np.loadtxt(spec_file,unpack=True)
        except IOError:
            tools.fatal_error("Spectrum File: "+spec_file+" does not exist at the specified location")
        mx = specData[mCol]
        xLog = specData[xCol]
        if not "ann" in mode:
            mxEff = phys.mx*0.5
        else:
            mxEff = phys.mx
        if sim.e_bins is None:
            spec_length = 51
            sim.e_bins = 51
        else:
            spec_length = sim.e_bins
        for ch,br in zip(phys.channel,phys.branching):
            chData = specData[chCols[ch]]
            if mxEff > max(mx) or mxEff < min(mx):
                tools.fatal_error("Required WIMP mass {} GeV does not lie within the data set found in {}".format(mxEff,spec_file))
            elif mxEff in mx:
                mIndices = np.where(mx==mxEff)
                eData = 10**(xLog[mIndices])*mxEff/phys.me
                dnData = chData[mIndices]/np.log(10.0)/10**(xLog[mIndices])/mxEff*phys.me
                intSpec = interp1d(eData,dnData)
                checkSpectra(spec_file,phys,np.logspace(np.log10(eData[0]*1.00001),np.log10(eData[-1]*0.99999),num=spec_length)) #makes sure domain is correct
                if "positrons" in spec_file:
                    newE = phys.spectrum[0]
                    dnData = intSpec(newE)
                    phys.spectrum[1] += dnData*br
                elif "gamma" in spec_file:
                    newE = phys.gamma_spectrum[0]
                    dnData = intSpec(newE)
                    phys.gamma_spectrum[1] += dnData*br
                elif "neutrino" in spec_file:
                    newE = phys.nu_spectrum[0]
                    dnData = intSpec(newE)
                    phys.nu_spectrum[1] += dnData*br
            else:
                eData,dnData = interpolateInputXDomain(mxEff,mx,xLog,chData,spec_length)
                eData *= 1.0/phys.me
                dnData *= phys.me
                checkSpectra(spec_file,phys,eData)
                #print(phys.spectrum)
                if "positrons" in spec_file:
                    phys.spectrum[1] += dnData*br
                elif "gamma" in spec_file:
                    phys.gamma_spectrum[1] += dnData*br
                elif "neutrino" in spec_file:
                    phys.nu_spectrum[1] += dnData*br
    else:
        mCol = 0
        xCol = 1
        nCol = 2
        if not "ann" in mode:
            mxEff = phys.mx*0.5
        else:
            mxEff = phys.mx
        try:
            specData = np.loadtxt(spec_file,unpack=True)
        except IOError:
            tools.fatal_error("Spectrum File: "+spec_file+" does not exist at the specified location")
        mx = specData[mCol]
        xLog = specData[xCol]
        chData = specData[nCol]
        if sim.e_bins is None:
            spec_length = 51
            sim.e_bins = 51
        else:
            spec_length = sim.e_bins

        if mxEff > max(mx) or mxEff < min(mx):
            tools.fatal_error("Required WIMP mass {} GeV does not lie within the data set found in {}".format(mxEff,spec_file))
        elif mxEff in mx:
            mIndices = np.where(mx==mxEff)
            eData = 10**(xLog[mIndices])*mxEff/phys.me
            dnData = chData[mIndices]/np.log(10.0)/10**(xLog[mIndices])/mxEff*phys.me
            intSpec = interp1d(eData,dnData)
            checkSpectra(spec_file,phys,np.logspace(np.log10(eData[0]*1.00001),np.log10(eData[-1]*0.99999),num=spec_length)) #makes sure domain is correct
            if "positrons" in spec_file:
                newE = phys.spectrum[0]
                dnData = intSpec(newE)
                phys.spectrum[1] = dnData
            elif "gamma" in spec_file:
                newE = phys.gamma_spectrum[0]
                dnData = intSpec(newE)
                phys.gamma_spectrum[1] = dnData
            elif "neutrino" in spec_file:
                newE = phys.nu_spectrum[0]
                dnData = intSpec(newE)
                phys.nu_spectrum[1] = dnData
        else:
            specDomain = checkSpecDomain(specData)
            if specDomain is None:
                tools.fatal_error("Interpolation is not supported for custom models where either energy or x domains do not match for every mass")
            elif specDomain == "e":
                eData = 10**(xLog)*mx
                dnData = chData/np.log(10.0)/10**(xLog)/mx
                eData,dnData = interpolateInputEDomain(mxEff,mx,eData,dnData,spec_length)
                eData *= 1.0/phys.me
                dnData *= phys.me
                checkSpectra(spec_file,phys,eData) #makes sure domain is correct
                #print(phys.spectrum)
                if "positrons" in spec_file:
                    phys.spectrum[1] = dnData
                elif "gamma" in spec_file:
                    phys.gamma_spectrum[1] = dnData
                elif "neutrino" in spec_file:
                    phys.nu_spectrum[1] = dnData
            elif specDomain == "x":
                eData,dnData = interpolateInputXDomain(mxEff,mx,xLog,chData,spec_length)
                eData *= 1.0/phys.me
                dnData *= phys.me
                checkSpectra(spec_file,phys,eData) #makes sure domain is correct
                #print(phys.spectrum)
                if "positrons" in spec_file:
                    phys.spectrum[1] = dnData
                elif "gamma" in spec_file:
                    phys.gamma_spectrum[1] = dnData
                elif "neutrino" in spec_file:
                    phys.nu_spectrum[1] = dnData

def checkSpectra(spec_file,phys,eData):
    if "positrons" in spec_file:
        specTarget = "spectrum"
        specMin = "specMin"
        specMax = "specMax"
    elif "gamma" in spec_file:
        specTarget = "gamma_spectrum"
        specMin = "gamma_specMin"
        specMax = "gamma_specMax"
    elif "neutrino" in spec_file:
        specTarget = "nu_spectrum"
        specMin = "nu_specMin"
        specMax = "nu_specMax"
    if getattr(phys,specTarget)[0] is None and getattr(phys,specTarget)[1] is None:
        setattr(phys,specTarget,[eData,np.zeros(len(eData))])
        setattr(phys,specMin,eData[0])
        setattr(phys,specMax,eData[-1])

def checkSpecDomain(specData):
    xData = specData[1]
    mxData = specData[0]
    uMx = np.unique(mxData)
    mXData1 = xData[np.where(mxData==uMx[0])]
    mXData2 = xData[np.where(mxData==uMx[-1])]
    if len(mXData2) == len(mXData1):
        eDev = np.sum(10**(mXData1)*uMx[0]/10**(mXData2)/uMx[-1])/len(mXData1)
        xDev = np.sum(mXData1/mXData2)/len(mXData1)
        if eDev == 1.0:
            return "e"
        elif xDev == 1.0:
            return "x"
        else:
            return None
    else:
        return None

def interpolateInputXDomain(mx,mxSet,xLog,chData,specLength):
    chTable = []
    xTarget = np.linspace(min(xLog),max(xLog),num=specLength)
    for m in np.unique(mxSet):
        mData = chData[np.where(mxSet==m)]
        eData = xLog[np.where(mxSet==m)]
        intSpec = interp1d(eData,mData)
        chTable.append(intSpec(xTarget))
    #print(interp2d(xTarget,unique(mxSet),np.array(chTable))(xTarget,mx))
    return 10**(xTarget)*mx,interp2d(xTarget,np.unique(mxSet),np.array(chTable))(xTarget,mx)/np.log(10.0)/10**(xTarget)/mx

def interpolateInputEDomain(mx,mxSet,eData,dNData,specLength):
    uEData = eData[np.where(mxSet==mxSet[0])]
    uMxData = np.unique(mxSet)
    intp = interp2d(uEData,uMxData,dNData.reshape((len(uMxData),len(uEData))))
    eTarget = np.logspace(np.log10(np.unique(eData)[0]),np.log10(np.unique(eData)[-1]),num=specLength)
    return eTarget,intp(eTarget,mx)


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
    if phys.model_independent:
        if mode == "ann":
            for ch,br in zip(phys.channel,phys.branching):
                pos = join(spec_dir,"pos_"+ch+"_"+str(int(phys.mx))+"GeV.data")
                gamma = join(spec_dir,"gamma_"+ch+"_"+str(int(phys.mx))+"GeV.data")
                read_spectrum(pos,0,br,phys,sim)
                read_spectrum(gamma,1,br,phys,sim)
        elif mode == "decay":
            for ch,br in zip(phys.channel,phys.branching):
                pos = join(spec_dir,"pos_"+ch+"_"+str(int(phys.mx*0.5))+"GeV.data")
                gamma = join(spec_dir,"gamma_"+ch+"_"+str(int(phys.mx*0.5))+"GeV.data")
                read_spectrum(pos,0,br,phys,sim)
                read_spectrum(gamma,1,br,phys,sim)
    else:
        if int(phys.mx) == phys.mx:
            mStr = str(int(phys.mx))
        else:
            mStr = str(phys.mx)
        pos = join(spec_dir,"pos_"+phys.particle_model+"_"+mStr+"GeV.data")
        gamma = join(spec_dir,"gamma_"+phys.particle_model+"_"+mStr+"GeV.data")
        read_spectrum(pos,0,1.0,phys,sim)
        read_spectrum(gamma,1,1.0,phys,sim)

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
    if phys.model_independent:
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
                E_set = np.zeros(len(s),dtype=float)  #Energies set
                Q_set = np.zeros(len(s),dtype=float)  #electron generation function dn/dE
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
                    nu_spec = [None,np.zeros(sim.e_bins)]
                    nu_spec[0] = np.logspace(np.log10(phys.nu_specMin*1.00001),np.log10(phys.nu_specMax*0.99999),num=sim.e_bins)
                    phys.nu_spectrum[0] = nu_spec[0]
                    phys.nu_spectrum[1] = np.zeros(len(phys.nu_spectrum[0]))
                intSpec = interp1d(E_set,Q_set)
                newE = phys.nu_spectrum[0]
                Q_set = intSpec(newE)
                phys.nu_spectrum[1] += Q_set*br
    else:
        nu_e = join(spec_dir,"nu_e_"+phys.particle_model+"_"+str(phys.mx)+"GeV.data")
        nu_mu = join(spec_dir,"nu_mu_"+phys.particle_model+"_"+str(phys.mx)+"GeV.data")
        nu_tau = join(spec_dir,"nu_tau_"+phys.particle_model+"_"+str(phys.mx)+"GeV.data")
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
            E_set = np.zeros(len(s),dtype=float)  #Energies set
            Q_set = np.zeros(len(s),dtype=float)  #electron generation function dn/dE
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
                nu_spec = [None,np.zeros(sim.e_bins)]
                nu_spec[0] = np.logspace(np.log10(phys.nu_specMin*1.00001),np.log10(phys.nu_specMax*0.99999),num=sim.e_bins)
                phys.nu_spectrum[0] = nu_spec[0]
                phys.nu_spectrum[1] = np.zeros(len(phys.nu_spectrum[0]))
            intSpec = interp1d(E_set,Q_set)
            newE = phys.nu_spectrum[0]
            Q_set = intSpec(newE)
            phys.nu_spectrum[1] += Q_set
    phys.nu_flavour = flavour

def console_mode(phys,sim,halo,cosmo,loop):
    print("Dark matters is now in console mode")
    print("Enter a command: ")
    command_line = ""
    while not command_line.strip().lower == "exit": 
        command_line = input("> ")
        process_command(command_line,phys,sim,halo,cosmo,loop)


if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        in_file = args[0]
        test = open(in_file,"r")
        console_flag = False
        test.close()
    except IndexError:
        console_flag = True
    except IOError:
        tools.fatal_error("Invalid script file path {} supplied".format(in_file))


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