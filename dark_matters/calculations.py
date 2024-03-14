from genericpath import isfile
import numpy as np
from os.path import join
import os
from scipy.integrate import simpson
from astropy import constants,units
import sympy
from sympy.utilities.lambdify import lambdify

from .output import fatal_error,warning,calc_write,wimp_write,spacer_length
from .dictionary_checks import check_cosmology,check_calculation,check_diffusion,check_gas,check_halo,check_magnetic,check_particles
from .emissions import adi_electron,green_electron,fluxes,emissivity

def get_index(set,val):
    """
    Returns the index of an object inside an array, no checks are done, use with caution

    Arguments
    ---------------------------
    set : array-like
        Full set of values to search
    val : scalar
        Value to find index of

    Returns
    ---------------------------
    index : int
        Index of val inside set
    """
    return np.where(set==val)[0][0]

def take_samples(xmin,xmax,nx,spacing="log",axis=-1):
    """
    Wrapper method for using linspace or logspace without writing log10 everywhere

    Arguments
    ---------------------------
    xmin : float
        Starting value
    xmax : float
        Ending value
    nx : int
        Number of samples
    spacing : str, optional 
        "log" or "lin, chooses logspace or linspace (default is "log")

    Returns
    ---------------------------
    samples : array-like (nx)
        Values spaced between xmin and max
    """
    if spacing == "log":
        return np.logspace(np.log10(xmin),np.log10(xmax),num=nx,axis=-1)
    else:
        return np.linspace(xmin,xmax,num=nx,axis=-1)

def physical_averages(rmax,mode_exp,calc_data,halo_data,mag_data,gas_data):
    """
    Computes averages for magnetic field and halo data within radius rmax

    Arguments
    ---------------------------
    rmax : float
        Integration limit
    mode_exp  : float
        Flag, 1 for decay or 2 for annihilation
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties
    mag_data : dictionary
        Magnetic field
    gas_data : dictionary
        Gas distribution

    Returns
    ---------------------------
    b_av, ne_av : float,float
        Volume averages
    """
    def weighted_vol_avg(y,r,w=None):
        if w is None:
            w = np.ones_like(r)
        return simpson(y*w*r**2,r)/simpson(w*r**2,r)
    r_set = take_samples(halo_data['scale']*10**calc_data['log10_r_sample_min_factor'],rmax,100)
    if halo_data['halo_weights'] == "rho":
        weights = halo_data['halo_density_func'](r_set)**mode_exp #the average is weighted
    else:
        weights = np.ones_like(r_set)
    return weighted_vol_avg(mag_data['mag_field_func'](r_set),r_set,w=weights),weighted_vol_avg(gas_data['gas_density_func'](r_set),r_set,w=weights)


def calc_electrons(mx,calc_data,halo_data,part_data,mag_data,gas_data,diff_data,over_write=True):
    """
    Computes equilibrium electron distributions from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties
    part_data : dictionary
        Particle physics
    mag_data : dictionary
        Magnetic field
    gas_data : dictionary
        Das distribution
    diff_data : dictionary
        Diffusion properties
    over_write : boolean
        If True will replace any existing values in calc_data['results']

    Returns
    ---------------------------
    calc_data : dictionary
        Calculation information with electron distribution in calc_data['results']['electron_data']
    """
    m_index = get_index(calc_data['m_wimp'],mx)
    if (not calc_data['results']['electron_data'][m_index] is None) and (not over_write):
        print("="*spacer_length)
        print(f"Electron Equilibrium distribution exists for WIMP mass {mx} GeV and over_write=False, skipping")
        print("="*spacer_length)
        print("Process Complete")
        return calc_data
    if part_data['em_model'] == "annihilation":
        mode_exp = 2.0
    else:
        mode_exp = 1.0
    if part_data['decay_input']:
        mx_eff = mx
    else:
        mx_eff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mx_eff = mx
    if diff_data['loss_only']:
        diff = 0
        delta = 0
        d0 = diff_data['diff_constant']
    else:
        diff = 1
        delta = diff_data['diff_index']
        d0 = diff_data['diff_constant']
    if diff_data['diff_rmax'] == "2*Rvir":
        r_limit = 2*halo_data['rvir']
    else:
        r_limit = diff_data['diff_rmax']

    if "green" in calc_data['electron_mode']:
        b_av,ne_av = physical_averages(halo_data['green_averaging_scale'],mode_exp,calc_data,halo_data,mag_data,gas_data)
        if "gas_average_density" in gas_data.keys():
            ne_av = gas_data['gas_average_density']
        if "mag_field_average" in mag_data.keys():
            b_av = gas_data['mag_field_average']
    elif "green_averaging_scale" in halo_data.keys():
        halo_data.pop('green_averaging_scale')
        
    if part_data['em_model'] == "annihilation":
        sigv = part_data['cross_section']
    else:
        sigv = part_data['decay_rate']

    r_sample = take_samples(halo_data['scale']*10**calc_data['log10_r_sample_min_factor'],r_limit,calc_data['r_sample_num'])
    rho_dm_sample = halo_data['halo_density_func'](r_sample)
    b_sample = mag_data['mag_field_func'](r_sample)
    ne_sample = gas_data['gas_density_func'](r_sample)

    if "green" in calc_data['electron_mode']:
        #Note sigv is left out here to simplify numerical convergence, it is restored below
        E_set = take_samples(np.log10(calc_data['e_sample_min']/mx_eff),0,calc_data['e_sample_num'],spacing="lin")
        Q_set = part_data['d_ndx_interp']['positrons'](mx_eff,E_set).flatten()/np.log(1e1)/10**E_set/mx_eff*(constants.m_e*constants.c**2).to("GeV").value
        E_set = 10**E_set*mx_eff/(constants.m_e*constants.c**2).to("GeV").value
    else:
        E_set = 10**take_samples(np.log10(calc_data['e_sample_min']/mx_eff),0,calc_data['e_sample_num'],spacing="lin")*mx_eff
        Q_set = part_data['d_ndx_interp']['positrons'](mx_eff,np.log10(E_set/mx_eff)).flatten()/np.log(1e1)/E_set*sigv

    if np.all(Q_set == 0.0):
        warning("At WIMP mass {mx} GeV dN/dE functions are zero at all considered energies!\nNote that in decay cases we sample mx_eff= 0.5*mx")

    print("="*spacer_length)
    print("Calculating Electron Equilibriumn Distributions")
    print("="*spacer_length)
    if calc_data['electron_mode'] == "green-python":
        print("Solution via: Green's function (python implementation)")
        print(f'Magnetic Field Average Strength: {b_av:.2e} micro Gauss')
        print(f'Gas Average Density: {ne_av:.2e} cm^-3')
        print(f'Averaging scale: {halo_data["green_averaging_scale"]:.2e} Mpc')
        calc_data['results']['electron_data'][m_index] = green_electron.equilibrium_electrons_grid_partial(calc_data['e_green_sample_num'],E_set,Q_set,calc_data['r_green_sample_num'],r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,halo_data['z'],delta,diff,d0,diff_data["photon_density"],calc_data['thread_number'],calc_data['image_number'])*sigv
    elif calc_data['electron_mode'] == "green-c":
        print("Solution via: Green's function (c++ implementation)")
        print(f'Magnetic Field Average Strength: {b_av:.2e} micro Gauss')
        print(f'Gas Average Density: {ne_av:.2e} cm^-3')
        print(f'Averaging scale: {halo_data["green_averaging_scale"]:.2e} Mpc')
        py_file = "temp_electrons_py.out"
        c_file = "temp_electrons_c.in"
        wd = os.getcwd()
        calc_data['results']['electron_data'][m_index] = green_electron.electrons_from_c(join(wd,py_file),join(wd,c_file),calc_data['electron_exec_file'],calc_data['e_green_sample_num'],E_set,Q_set,calc_data['r_green_sample_num'],r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,halo_data['z'],delta,diff,d0,diff_data['photon_density'],num_threads=calc_data['thread_number'],num_images=calc_data['image_number'])
        if calc_data['results']['electron_data'][m_index] is None:
            fatal_error(f"The electron executable {calc_data['electron_exec_file']} is not compiled or location not specified correctly")
        else:
            calc_data['results']['electron_data'][m_index] *= sigv
    elif calc_data['electron_mode'] == "adi-python":
        print("Solution via: ADI method (python implementation)")
        r = sympy.symbols('r')
        dBdr_sample = lambdify(r,sympy.diff(mag_data['mag_field_func'](r),r))(r_sample)
        if np.isscalar(dBdr_sample):
            dBdr_sample = dBdr_sample*np.ones_like(r_sample)
        adi_solver = adi_electron.adi_scheme(benchmark_flag=calc_data['adi_bench_mark_mode'],const_delta_t=calc_data['adi_delta_t_constant'])
        calc_data['results']['electron_data'][m_index] = adi_solver.solve_electrons(mx,halo_data['z'],E_set,r_sample,rho_dm_sample,Q_set,b_sample,dBdr_sample,ne_sample,halo_data['scale'],1.0,diff_data['diff_index'],u_ph=diff_data['photon_density'],diff0=diff_data['diff_constant'],delta_t_min=calc_data['adi_delta_t_min'],loss_only=diff_data['loss_only'],mode_exp=mode_exp,delta_ti=calc_data['adi_delta_ti'],max_t_part=calc_data['adi_max_steps'],delta_t_reduction=calc_data['adi_delta_t_reduction'])*(constants.m_e*constants.c**2).to("GeV").value
    print("Process Complete")
    return calc_data

def calc_radio_em(mx,calc_data,halo_data,part_data,mag_data,gas_data,diff_data):
    """
    Computes radio emissivities from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties
    part_data : dictionary
        Particle physics
    mag_data : dictionary
        Magnetic field
    gas_data : dictionary
        Das distribution
    diff_data : dictionary
        Diffusion properties

    Returns
    ---------------------------
    calc_data : dictionary
        Calculation information with radio emissivity in calc_data['results']['radio_em_data']
    """
    print("="*spacer_length)
    print("Calculating Radio Emissivity")
    print("="*spacer_length)
    if part_data['em_model'] == "annihilation":
        mode_exp = 2.0
    else:
        mode_exp = 1.0
    if diff_data['diff_rmax'] == "2*Rvir":
        r_limit = 2*halo_data['rvir']
    else:
        r_limit = diff_data['diff_rmax']
    if part_data['decay_input']:
        mx_eff = mx
    else:
        mx_eff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mx_eff = mx
    m_index = get_index(calc_data['m_wimp'],mx)
    r_sample = take_samples(halo_data['scale']*10**calc_data['log10_r_sample_min_factor'],r_limit,calc_data['r_sample_num'])
    f_sample = calc_data['f_sample_values']
    x_sample = take_samples(np.log10(calc_data['e_sample_min']/mx_eff),0,calc_data['e_sample_num'],spacing="lin")
    g_sample = 10**x_sample*mx_eff/(constants.m_e*constants.c**2).to("GeV").value
    b_sample = mag_data['mag_field_func'](r_sample)
    ne_sample = gas_data['gas_density_func'](r_sample)
    if calc_data['results']['electron_data'][m_index] is None:
        calc_data = calc_electrons(mx,calc_data,halo_data,part_data,mag_data,gas_data,diff_data)
    electrons = calc_data['results']['electron_data'][m_index]
    if calc_data['results']['radio_em_data'][m_index] is None:
        calc_data['results']['radio_em_data'][m_index] = emissivity.radio_em_grid(electrons,f_sample,r_sample,g_sample,b_sample,ne_sample)
    print("Process Complete")
    return calc_data

def calc_primary_em(mx,calc_data,halo_data,part_data,diff_data):
    """
    Computes primary gamma-ray or neutrino emissivity from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties
    part_data : dictionary
        Particle physics

    Returns
    ---------------------------
    calc_data : dictionary
        Calculation information with emissivity in calc_data['results'][x], x = primary_em_data or neutrino_em_data
    """
    if calc_data['freq_mode'] in ["gamma","pgamma","all"]:
        print("="*spacer_length)
        print("Calculating Primary Gamma-ray Emissivity")
        print("="*spacer_length)
        spec_type = "gammas"
        em_type = 'primary_em_data'
    else:
        print("="*spacer_length)
        print("Calculating Neutrino Emissivity")
        print("="*spacer_length)
        spec_type = calc_data['freq_mode']
        em_type = 'neutrino_em_data'
    m_index = get_index(calc_data['m_wimp'],mx)
    if part_data['em_model'] == "annihilation":
        mode_exp = 2.0
    else:
        mode_exp = 1.0
    if diff_data['diff_rmax'] == "2*Rvir":
        r_limit = 2*halo_data['rvir']
    else:
        r_limit = diff_data['diff_rmax']
    if part_data['decay_input']:
        mx_eff = mx
    else:
        mx_eff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mx_eff = mx
    r_sample = take_samples(halo_data['scale']*10**calc_data['log10_r_sample_min_factor'],r_limit,calc_data['r_sample_num'])
    f_sample = calc_data['f_sample_values']
    x_sample = take_samples(np.log10(calc_data['e_sample_min']/mx_eff),0,calc_data['e_sample_num'],spacing="lin")
    if part_data['em_model'] == "annihilation":
        sigv = part_data['cross_section']
    else:
        sigv = part_data['decay_rate']
    g_sample = 10**x_sample*mx_eff/(constants.m_e*constants.c**2).to("GeV").value
    rho_sample = halo_data['halo_density_func'](r_sample)
    q_sample = part_data['d_ndx_interp'][spec_type](mx_eff,x_sample).flatten()/np.log(1e1)/10**x_sample/mx_eff*(constants.m_e*constants.c**2).to("GeV").value*sigv
    if np.all(q_sample == 0.0):
        warning("At WIMP mass {mx} GeV dN/dE functions are zero at all considered energies!\nNote that in decay cases we sample mx_eff= 0.5*mx")
    calc_data['results'][em_type][m_index] = emissivity.primary_em_high_e(mx,rho_sample,halo_data['z'],g_sample,q_sample,f_sample,mode_exp)
    print("Process Complete")
    return calc_data

def calc_secondary_em(mx,calc_data,halo_data,part_data,mag_data,gas_data,diff_data):
    """
    Computes secondary high-energy emissivities from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties
    part_data : dictionary
        Particle physics
    mag_data : dictionary
        Magnetic field
    gas_data : dictionary
        Das distribution
    diff_data : dictionary
        Diffusion properties

    Returns
    ---------------------------
    calc_data : dictionary
        Calculation information with emissivity in calc_data['results']['secondary_em_data']
    """
    print("="*spacer_length)
    print("Calculating Secondary Gamma-ray Emissivity")
    print("="*spacer_length)
    if part_data['em_model'] == "annihilation":
        mode_exp = 2.0
    else:
        mode_exp = 1.0
    m_index = get_index(calc_data['m_wimp'],mx)
    if diff_data['diff_rmax'] == "2*Rvir":
        r_limit = 2*halo_data['rvir']
    else:
        r_limit = diff_data['diff_rmax']
    if part_data['decay_input']:
        mx_eff = mx
    else:
        mx_eff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mx_eff = mx
    r_sample = take_samples(halo_data['scale']*10**calc_data['log10_r_sample_min_factor'],r_limit,calc_data['r_sample_num'])
    f_sample = calc_data['f_sample_values'] #frequency values
    x_sample = take_samples(np.log10(calc_data['e_sample_min']/mx_eff),0,calc_data['e_sample_num'],spacing="lin")
    g_sample = 10**x_sample*mx_eff/(constants.m_e*constants.c**2).to("GeV").value
    ne_sample = gas_data['gas_density_func'](r_sample)
    if calc_data['results']['electron_data'][m_index] is None:
        calc_data = calc_electrons(mx,calc_data,halo_data,part_data,mag_data,gas_data,diff_data)
    electrons = calc_data['results']['electron_data'][m_index]
    calc_data['results']['secondary_em_data'][m_index] = emissivity.secondary_em_high_e(electrons,halo_data['z'],g_sample,f_sample,ne_sample,diff_data['photon_temp'])
    print("Process Complete")
    return calc_data  

def calc_flux(mx,calc_data,halo_data,diff_data):
    """
    Computes flux from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties

    Returns
    ---------------------------
    calc_data : dictionary
        Calculation information with flux in calc_data['results']['final_data']
    """
    print("="*spacer_length)
    print("Calculating Flux")
    print("="*spacer_length)
    print(f"Frequency mode: {calc_data['freq_mode']}")
    if diff_data['diff_rmax'] == "2*Rvir":
        r_limit = 2*halo_data['rvir']
    else:
        r_limit = diff_data['diff_rmax']
    if 'rmax_integrate' in calc_data.keys():
        if calc_data['rmax_integrate'] == "Rvir":
            rmax = halo_data['rvir']
        elif calc_data['rmax_integrate'] == -1:
            rmax = r_limit
        else:
            rmax = calc_data['rmax_integrate']
        print(f"Integration radius: {rmax} Mpc")
    else:
        rmax = np.tan(calc_data['angmax_integrate']/180/60*np.pi)*halo_data['distance']/(1+halo_data['z'])**2
        print(f"Integration radius: {calc_data['angmax_integrate']} arcmins = {rmax} Mpc")
    m_index = get_index(calc_data['m_wimp'],mx)
    if calc_data['freq_mode'] == "all":
        emm = calc_data['results']['radio_em_data'][m_index] + calc_data['results']['primary_em_data'][m_index] + calc_data['results']['secondary_em_data'][m_index]
    elif calc_data['freq_mode'] == "gamma":
        emm = calc_data['results']['primary_em_data'][m_index] + calc_data['results']['secondary_em_data'][m_index]
    elif calc_data['freq_mode'] == "pgamma":
        emm = calc_data['results']['primary_em_data'][m_index]
    elif calc_data['freq_mode'] == "sgamma":
        emm = calc_data['results']['secondary_em_data'][m_index]
    elif calc_data['freq_mode'] == "radio":
        emm = calc_data['results']['radio_em_data'][m_index] 
    elif "neutrinos" in calc_data['freq_mode']:
        emm = calc_data['results']['neutrino_em_data'][m_index]
    r_sample = take_samples(halo_data['scale']*10**calc_data['log10_r_sample_min_factor'],r_limit,calc_data['r_sample_num'])
    f_sample = calc_data['f_sample_values']
    calc_data['results']['final_data'][m_index] = fluxes.flux_grid(rmax,halo_data['distance'],f_sample,r_sample,emm,boost_mod=1.0,ergs=calc_data["out_cgs"])
    print("Process Complete")
    return calc_data

def calc_sb(mx,calc_data,halo_data,diff_data):
    """
    Computes surface brightness from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties

    Returns
    ---------------------------
    calc_data : dictionary
        Calculation information with surface brightness in calc_data['results']['final_data']
    """
    print("="*spacer_length)
    print("Calculating Surface Brightness")
    print("="*spacer_length)
    print(f"Frequency mode: {calc_data['freq_mode']}")
    m_index = get_index(calc_data['m_wimp'],mx)
    if calc_data['freq_mode'] == "all":
        emm = calc_data['results']['radio_em_data'][m_index] + calc_data['results']['primary_em_data'][m_index] + calc_data['results']['secondary_em_data'][m_index]
    elif calc_data['freq_mode'] == "gamma":
        emm = calc_data['results']['primary_em_data'][m_index] + calc_data['results']['secondary_em_data'][m_index]
    elif calc_data['freq_mode'] == "pgamma":
        emm = calc_data['results']['primary_em_data'][m_index]
    elif calc_data['freq_mode'] == "sgamma":
        emm = calc_data['results']['secondary_em_data'][m_index]
    elif calc_data['freq_mode'] == "radio":
        emm = calc_data['results']['radio_em_data'][m_index] 
    elif "neutrinos" in calc_data['freq_mode']:
        emm = calc_data['results']['neutrino_em_data'][m_index]
    nu_SB = []
    if diff_data['diff_rmax'] == "2*Rvir":
        r_limit = 2*halo_data['rvir']
    else:
        r_limit = diff_data['diff_rmax']
    r_sample = take_samples(halo_data['scale']*10**calc_data['log10_r_sample_min_factor'],r_limit,calc_data['r_sample_num'])
    f_sample = calc_data['f_sample_values']
    for nu in f_sample:
        nu_SB.append(fluxes.surface_brightness_loop(nu,f_sample,r_sample,emm,ergs=calc_data["out_cgs"])[1])
    calc_data['results']['final_data'][m_index] = np.array(nu_SB)
    calc_data['results']['ang_sample_values'] = np.arctan(take_samples(halo_data['scale']*10**calc_data['log10_r_sample_min_factor'],r_limit,calc_data['r_sample_num'])/halo_data['distance']*(1+halo_data['z'])**2)/np.pi*180*60
    print("Process Complete")
    return calc_data  

def calc_j_flux(mx,calc_data,halo_data,part_data):
    """
    Computes J/D-factor flux from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties
    part_data : dictionary
        Particle physics

    Returns
    ---------------------------
    calc_data : dictionary
        Calculation information with flux in calc_data['results']['final_data']
    """
    print("="*spacer_length)
    print("Calculating Flux From J/D-factor")
    print("="*spacer_length)
    print(f"Frequency mode: {calc_data['freq_mode']}")

    if part_data['em_model'] == "annihilation":
        mode_exp = 2.0
        j_fac = halo_data['j_factor']
    else:
        mode_exp = 1.0
        j_fac = halo_data['d_factor']
    if part_data['decay_input']:
        mx_eff = mx
    else:
        mx_eff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mx_eff = mx
    if not calc_data['freq_mode'] == 'pgamma' or 'neutrinos' in calc_data['freq_mode']:
        fatal_error(f"jflux cannot be run with freq_mode={calc_data['freq_mode']}")
    if calc_data['freq_mode'] == "pgamma":
        spec_type = "gammas"
    else:
        spec_type = calc_data['freq_mode']
    m_index = get_index(calc_data['m_wimp'],mx)
    f_sample = calc_data['f_sample_values']
    x_sample = take_samples(np.log10(calc_data['e_sample_min']/mx_eff),0,calc_data['e_sample_num'],spacing="lin")
    g_sample = 10**x_sample*mx_eff/(constants.m_e*constants.c**2).to("GeV").value  
    q_sample_gamma = part_data['d_ndx_interp'][spec_type](mx_eff,x_sample).flatten()/np.log(1e1)/10**x_sample/mx_eff*(constants.m_e*constants.c**2).to("GeV").value*part_data['cross_section']
    calc_data['results']['final_data'][m_index] = fluxes.flux_from_j_factor(mx,halo_data['z'],j_fac,f_sample,g_sample,q_sample_gamma,mode_exp)
    print("Process Complete")
    return calc_data

def run_checks(calc_data,halo_data,part_data,mag_data,gas_data,diff_data,cosmo_data,clear):
    """
    Processes dictionaries to prepare for calculations, will crash if this is not possible
    Arguments
    ---------------------------
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties
    part_data : dictionary
        Particle physics
    mag_data : dictionary
        Magnetic field
    gas_data : dictionary
        Das distribution
    diff_data : dictionary
        Diffusion properties
    cosmo_data : dictionary
        Cosmology properties
    clear : string (optional)
        What results to clear, can be 'all', 'observables' or 'final' (defaults to 'all')
    Returns
    ---------------------------
    All given dictionaries checked and ready for calculations
    """
    cosmo_data = check_cosmology(cosmo_data)
    if not calc_data['calc_mode'] == "jflux":
        if (not calc_data['freq_mode'] == "pgamma") and (not "neutrinos" in calc_data['freq_mode']):
            mag_data = check_magnetic(mag_data)
            gas_data = check_gas(gas_data)
        diff_data = check_diffusion(diff_data)
        halo_data = check_halo(halo_data,cosmo_data)
    elif calc_data['calc_mode'] == "jflux" and (not 'j_factor' in halo_data.keys()) and part_data['em_model'] == "annihilation":
        halo_data = check_halo(halo_data,cosmo_data)
    elif calc_data['calc_mode'] == "jflux" and (not 'd_factor' in halo_data.keys()) and (not part_data['em_model'] == "annihilation"):
        halo_data = check_halo(halo_data,cosmo_data)
    else:
        halo_data = check_halo(halo_data,cosmo_data,minimal=True)
    calc_data = check_calculation(calc_data)
    if clear == "all":
        calc_data['results'] = {'electron_data':[],'radio_em_data':[],'primary_em_data':[],'secondary_em_data':[],'final_data':[],'neutrino_em_data':[]}
        for i in range(len(calc_data['m_wimp'])):
            calc_data['results']['electron_data'].append(None)
            calc_data['results']['radio_em_data'].append(None)
            calc_data['results']['primary_em_data'].append(None)
            calc_data['results']['secondary_em_data'].append(None)
            calc_data['results']['neutrino_em_data'].append(None)
            calc_data['results']['final_data'].append(None)
    elif clear == "observables":
        if 'results' in calc_data.keys():
            if 'electron_data' in calc_data['results'].keys():
                if np.any(calc_data['results']['electron_data'] is None):
                    fatal_error("calculations.run_checks(): You cannot run with clear=observables if your calc_data dictionary has incomplete electron_data")
            else:
                fatal_error("calculations.run_checks(): You cannot run with clear=observables if your calc_data dictionary has no existing electron_data")
        else:
            fatal_error("calculations.run_checks(): You cannot run with clear=observables if your calc_data dictionary has no existing results")
        calc_data['results']['radio_em_data'] = []
        calc_data['results']['primary_em_data'] = []
        calc_data['results']['final_data'] = []
        calc_data['results']['secondary_em_data'] = []
        calc_data['results']['neutrino_em_data'] = []
        for i in range(len(calc_data['m_wimp'])):
            calc_data['results']['radio_em_data'].append(None)
            calc_data['results']['primary_em_data'].append(None)
            calc_data['results']['secondary_em_data'].append(None)
            calc_data['results']['neutrino_em_data'].append(None)
            calc_data['results']['final_data'].append(None)
    else:
        if 'results' in calc_data.keys():
            if 'electron_data' in calc_data['results'].keys():
                if np.any(calc_data['results']['electron_data'] is None):
                    fatal_error("calculations.run_checks(): You cannot run with clear=final if your calc_data dictionary has incomplete electron_data")
            else:
                fatal_error("calculations.run_checks(): You cannot run with clear=final if your calc_data dictionary has no existing electron_data")
        else:
            fatal_error("calculations.run_checks(): You cannot run with clear=observables if your calc_data dictionary has no existing results")
        calc_data['results']['final_data'] = []
        for i in range(len(calc_data['m_wimp'])):
            calc_data['results']['final_data'].append(None)
    result_units = {"electron_data":"GeV/cm^3","radio_em_data":"GeV/cm^3","primary_em_data":"GeV/cm^3","secondary_em_data":"GeV/cm^3","f_sample_values":"MHz"}
    if "flux" in calc_data['calc_mode']:
        if calc_data['out_cgs']:
            result_units['final_data'] = "erg/(cm^2 s)"
        else:
            result_units['final_data'] = "Jy"
    else:
        if calc_data['out_cgs']:
            result_units['final_data'] = "erg/(cm^2 s arcmin^2)"
        else:
            result_units['final_data'] = "Jy/arcmin^2"
        result_units['ang_sample_values'] = "arcmin"
    calc_data['results']['units'] = result_units
    part_data = check_particles(part_data,calc_data)
    return calc_data,halo_data,part_data,mag_data,gas_data,diff_data,cosmo_data

def run_calculation(calc_data,halo_data,part_data,mag_data,gas_data,diff_data,cosmo_data,over_write_electrons=True,clear="all"):
    """
    Processes dictionaries and runs the requested operations

    Arguments
    ---------------------------
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties
    part_data : dictionary
        Particle physics
    mag_data : dictionary
        Magnetic field
    gas_data : dictionary
        Das distribution
    diff_data : dictionary
        Diffusion properties
    diff_data : dictionary
        Diffusion properties
    cosmo_data : dictionary
        Cosmology properties
    over_write_electrons : boolean
        if False will not overwrite existing electron_data values
    clear : string
        What results to clear, can be 'all', 'observables' or 'final'

    Returns
    ---------------------------
    All given dictionaries checked and updated, including calc_data with completed calc_data['results']
    """
    
    calc_data,halo_data,part_data,mag_data,gas_data,diff_data,cosmo_data = run_checks(calc_data,halo_data,part_data,mag_data,gas_data,diff_data,cosmo_data,clear)
    print("="*spacer_length)
    print("Beginning DarkMatters calculations")
    print("="*spacer_length)
    print(f"Frequency mode: {calc_data['freq_mode']}")
    print(f"Calculation type: {calc_data['calc_mode']}")
    if calc_data['calc_mode'] == "jflux":
        calc_j_flag = False
        if not 'j_factor' in halo_data.keys() and part_data['em_model'] == "annihilation":
            calc_j_flag = True
            mode_exp = 2.0
            unit_fac = (1*units.Unit("Msun^2/Mpc^5")*constants.c**4).to("GeV^2/cm^5").value
        elif not 'd_factor' in halo_data.keys() and part_data['em_model'] == "decay":
            calc_j_flag = True
            mode_exp = 1.0
            unit_fac = (1*units.Unit("Msun/Mpc^2")*constants.c**2).to("GeV/cm^2").value
        if calc_j_flag:
            theta_min = np.arctan(halo_data['scale']*1e-3/halo_data['distance'])
            if 'angmax_integrate' in calc_data.keys():
                theta_max = calc_data['angmax_integrate']
            else:
                theta_max = np.arctan(calc_data['rmax_integrate']/halo_data['distance'])
            if not 'truncation_scale' in halo_data.keys():
                rt = halo_data['rvir']
            else:
                rt = halo_data['truncation_scale']
            rt_check = np.arctan(rt/halo_data['distance']) < theta_max
            if rt_check:
                fatal_error("'angmax_integrate' or 'rmax_integrate' is larger than the 'truncation_scale' (defaults to virial radius)")
            j_fac = 10**fluxes.get_j_factor(theta_max,theta_min,halo_data['distance'],halo_data['halo_density_func'],mode_exp,rt)*unit_fac
            print(j_fac)
            if part_data['em_model'] == "annihilation":
                halo_data['j_factor'] = j_fac
            else:
                halo_data['d_factor'] = j_fac
        calc_write(calc_data,halo_data,part_data,mag_data,gas_data,diff_data)
    for mx in calc_data['m_wimp']:
        wimp_write(mx,part_data)
        m_index = get_index(calc_data['m_wimp'],mx)
        if not calc_data['calc_mode'] == "jflux":
            if (not calc_data['freq_mode'] == "pgamma") and (not "neutrinos" in calc_data['freq_mode']):
                calc_data = calc_electrons(mx,calc_data,halo_data,part_data,mag_data,gas_data,diff_data,over_write=over_write_electrons)
            if calc_data['freq_mode'] in ["all","radio"]:
                calc_data = calc_radio_em(mx,calc_data,halo_data,part_data,mag_data,gas_data,diff_data)
            if calc_data['freq_mode'] in ["all","gamma","pgamma"]:
                calc_data = calc_primary_em(mx,calc_data,halo_data,part_data,diff_data)
            if calc_data['freq_mode'] in ["all","gamma","sgamma"]:
                calc_data = calc_secondary_em(mx,calc_data,halo_data,part_data,mag_data,gas_data,diff_data)
            if "neutrinos" in calc_data['freq_mode']:
                calc_data = calc_primary_em(mx,calc_data,halo_data,part_data,diff_data)
            if calc_data['calc_mode'] == "flux":
                calc_data = calc_flux(mx,calc_data,halo_data,diff_data)
            elif calc_data['calc_mode'] == "sb":
                calc_data = calc_sb(mx,calc_data,halo_data,diff_data)
            if diff_data['diff_rmax'] == "2*Rvir":
                r_limit = 2*halo_data['rvir']
            else:
                r_limit = diff_data['diff_rmax']
            calc_data['results']['r_sample_values'] = take_samples(halo_data['scale']*10**calc_data['log10_r_sample_min_factor'],r_limit,calc_data['r_sample_num'])
            calc_data['results']['units']['r_sample_values'] = "Mpc"
            if part_data['em_model'] == "annihilation":
                mode_exp = 2.0
            else:
                mode_exp = 1.0
            if calc_data['freq_mode'] in ["all","gamma","sgamma","radio"]:
                if part_data['decay_input']:
                    mx_eff = mx
                else:
                    mx_eff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mx_eff = mx
                x_sample = take_samples(np.log10(calc_data['e_sample_min']/mx_eff),0,calc_data['e_sample_num'],spacing="lin")
                calc_data['results']['e_sample_values'] = 10**x_sample*mx_eff
                calc_data['results']['units']['e_sample_values'] = "GeV"
        else:
            calc_data = calc_j_flux(mx,calc_data,halo_data,part_data)
    calc_data['results']['f_sample_values'] = calc_data['f_sample_values']
    py_file = "temp_electrons_py.out"
    c_file = "temp_electrons_c.in"
    wd = os.getcwd()
    if isfile(join(wd,py_file)):
        os.remove(join(wd,py_file))
    if isfile(join(wd,c_file)):
        os.remove(join(wd,c_file))
    return {'calc_data':calc_data,'halo_data':halo_data,'part_data':part_data,'mag_data':mag_data,'gas_data':gas_data,'diff_data':diff_data,'cosmo_data':cosmo_data}
