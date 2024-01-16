"""
DarkMatters module for handling input
"""
import numpy as np
import yaml,json
from astropy import units
from scipy.interpolate import interp2d
import os
from .output import fatal_error,check_quant,warning

def get_spectral_data(spec_dir,part_model,spec_set,mode="annihilation"):
    """
    Retrieves particle yield spectra for a given model, set of WIMP masses, and products

    Arguments
    ---------------------------
    spec_dir : str
        Path of folder where spectra are stored
    part_model : str 
        Label of particle physics model
    spec_set :  str, list
        Particle yield spectra to be loaded. Allowed entries are "gammas", "positrons", "neutrinos_x" where x = mu,e, or tau
    mode : str, optional 
        Annihilation or decay
    pppcdb4dm : bool
        Flag for using pppc4dmid tables

    Returns
    ---------------------------
    spec_dict : dictionary
        Dictionary of yield spectra, keys matching spec_set, values are interpolating functions
    """
    spec_dict = {}
    for f in spec_set:
        if part_model in ["bb","qq","ww","ee","hh","tautau","mumu","tt","zz"]:
            spec_dict[f] = read_spectrum(os.path.join(spec_dir,f"AtProduction_{f}.dat"),part_model,mode=mode,pppc4dmid=True)
        else:
            spec_dict[f] = read_spectrum(os.path.join(spec_dir,f"{part_model}_AtProduction_{f}.dat"),part_model,mode=mode,pppc4dmid=False)
    return spec_dict

def read_spectrum(spec_file,part_model,mode="annihilation",pppc4dmid=True):
    """
    Reads file to get particle yield spectra for a given model and set of WIMP masses

    Arguments
    ---------------------------
    spec_file : str
        Path of spectrum file
    part_model : str 
        Label of particle physics model
    mode : float, optional 
        Flag, 2.0 for annihilation or 1.0 for decay
    pppc4dmid : bool, optional 
        Flag for using PPPC4DMID tables

    Returns
    ---------------------------
    intp: interpolating function (mx,log10(energy/mx))
        Interpolating function for particle yields

    Notes
    ---------------------------
    file names format : "AtProduction_part_model_products.dat", "products" can be "positrons", "gammas", or "neutrinos_e" etc 
    A custom spec_file must be formatted as follows:
    column 0: WIMP mass in GeV, column 1: log10(energy/mx) , column 2: dN/dlog10(energy/mx)
    """
    #mDM      Log[10,x]   eL         eR         e          \[Mu]L     \[Mu]R     \[Mu]      \[Tau]L    \[Tau]R    \[Tau]     q            c            b            t            WL          WT          W           ZL          ZT          Z           g            \[Gamma]    h           \[Nu]e     \[Nu]\[Mu]   \[Nu]\[Tau]   V->e       V->\[Mu]   V->\[Tau]
    ch_cols = {"ee":4,"mumu":7,"tautau":10,"qq":11,"bb":13,"tt":14,"ww":17,"zz":20,"gamma":22,'hh':23}
    if pppc4dmid:
        n_col = ch_cols[part_model]
    else:
        n_col = 2
    m_col = 0
    x_col = 1
    try:
        spec_data = np.loadtxt(spec_file,unpack=True)
    except IOError:
        fatal_error("Spectrum File: "+spec_file+" does not exist at the specified location")
    mx = np.unique(spec_data[m_col])
    x_log = np.unique(spec_data[x_col])
    dn_data = spec_data[n_col]
    #dn_data.reshape((len(mx),len(x_log)))
    if mode == "annihilation":
        intp = interp2d(mx,x_log,dn_data,fill_value=0.0)
    else:
        intp = interp2d(mx,x_log,dn_data,fill_value=0.0)
    return intp    

def read_input_file(input_file,in_mode="yaml"):
    """
    Reads a yaml file and builds dictionaries 

    Arguments
    ---------------------------
    input_file : str 
        Path of input file

    Returns
    ---------------------------
    data_sets : dictionaries
        Dictionaries storing information on: calculations, halo properties, particle physics, magnetic fields, gas distribution, diffusion, and cosmology

    Notes
    ---------------------------
    All dictionaries are returned, empty dictionaries indicate no properties were set in the file
    """
    stream = open(input_file, 'r')
    if in_mode == "yaml":
        input_data = yaml.load(stream,Loader=yaml.SafeLoader)
    elif in_mode == "json":
        input_data = json.load(stream)
    else:
        fatal_error(f"The argument in_mode = {in_mode} given to input.readinput_file() does not match any valid input modes")
    stream.close()
    valid_keys = ["halo_data","mag_data","gas_data","diff_data","part_data","calc_data","cosmo_data"]
    dm_units = {"temperature":"K","energy_density":"eV/cm^3","decay_rate":"1/s","cross_section":"cm^3/s","time":"yr","distance":"Mpc","mass":"solMass","density":"Msun/Mpc^3","num_density":"1/cm^3","magnetic":"microGauss","energy":"GeV","frequency":"MHz","angle":"arcmin","j_factor":"GeV^2/cm^5","d_factor":"GeV/cm^2","diff_constant":"cm^2/s"}
    data_sets = {}
    for key in valid_keys:
        data_sets[key] = {}
    for h in input_data.keys():
        if not h in valid_keys:
            fatal_error(f"The key {h} in the file {input_file} is not valid, options are {valid_keys}")
        for x in input_data[h].keys():
            if not isinstance(input_data[h][x],dict):
                data_sets[h][x] = input_data[h][x]
            elif 'unit' in input_data[h][x].keys():
                quant = check_quant(x) #we find out what kind of units x has, i.e. distance, mass etc
                if not quant is None:
                    unit_str = dm_units[quant] #get the unit DM uses internally
                else:
                    fatal_error(f"{h} property {x} does not accept a unit argument")
                try:
                    data_sets[h][x] = ((input_data[h][x]['value']*units.Unit(input_data[h][x]['unit'])).to(unit_str)).value #convert the units to internal system
                except AttributeError:
                    data_sets[h][x] = (input_data[h][x]['value']*units.Unit(input_data[h][x]['unit'])).to(unit_str)
                except:
                    fatal_error(f"Processing failed on {h} property {x} ")
    if len(data_sets['mag_data']) > 0:
        data_sets['mag_data']['mag_func_lock'] = False
    return data_sets

def read_dm_output(f_name,in_mode="yaml"):
    """
    Reads in an output yaml file created by DarkMatters

    Arguments
    ---------------------------
    f_name : str 
        Path of file

    Returns
    ---------------------------
    Dictionaries storing information on: calculations, halo properties, particle physics, magnetic fields, gas distribution, diffusion, and cosmology
    """
    try:
        stream = open(f_name, 'r')
    except IOError:
        fatal_error(f"File {f_name} not found")
    if in_mode == "yaml":
        try:
            in_data = yaml.load(stream,Loader=yaml.SafeLoader)
        except:
            stream.close()
            stream = open(f_name, 'r')
            warning(f"Loading {f_name} with unsafeLoader (probably due to numpy objects)")
            in_data = yaml.load(stream,Loader=yaml.UnsafeLoader)
    elif in_mode == "json":
        in_data = json.load(stream)
    else:
        fatal_error(f"The argument in_mode = {in_mode} given to input.readinput_file() does not match any valid input modes")
    stream.close()
    return in_data
