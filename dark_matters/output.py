"""
DarkMatters module for handling output
"""
import numpy as np
import sys,yaml,json,os
from scipy.interpolate import interp1d,RegularGridInterpolator
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy import units,constants,wcs

spacer_length = 55


def check_quant(key):
    """
    Finds unit for a given quantity
    
    Arguments
    ---------------------------
    key : string
        Property to find unit for

    Returns
    ---------------------------
    h : string
        Unit in astropy format or None, if not found
    """
    in_file =open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"config/quantities.yaml"),"r")
    quant_dict = yaml.load(in_file,Loader=yaml.SafeLoader)
    in_file.close()
    for h in quant_dict:
        if key in quant_dict[h]:
            return h
    return None

def fatal_error(err_string):
    """
    Display error string and exit program

    Arguments
    ---------------------------
    err_string: string 
        Error string to be displayed

    Returns
    ---------------------------
    None
    """
    print("#"*spacer_length)
    print("                   Fatal Error")
    print("#"*spacer_length)
    raise SystemExit(err_string)

def warning(err_string):
    """
    Display error string

    Arguments
    ---------------------------
    err_string: string 
        Error string to be displayed

    Returns
    ---------------------------
    None
    """
    print("*"*spacer_length)
    print("                   Warning")
    print("*"*spacer_length)
    print(err_string)
    print("*"*spacer_length)

def get_calc_id(calc_data,halo_data,part_data,diff_data,tag=None):
    """
    Builds an output file id code

    Arguments
    ---------------------------
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties
    part_data : dictionary
        Particle physics properties
    diff_data : dictionary
        Diffusion properties
    tag : string (optional)
        String added to file ID

    Returns
    ---------------------------
    file_id : string
        File name for calculations
    """
    dm_str = halo_data['profile']
    if 'index' in halo_data.keys():
        dm_str += f"-{halo_data['index']:.2f}"
    dm_str += "_"

    if not calc_data['calc_mode'] == "jflux":
        if not "green" in calc_data['electron_mode']:
            w_str = ""
        elif halo_data['halo_weights'] == "flat":
            w_str = "weights-flat_"
        else:
            w_str = "weights-rho_"

        if halo_data['distance'] < 0.1 or halo_data['distance'] > 1e3:
            dist_str = f"dl-{halo_data['distance']:.2e}Mpc_"
        elif  halo_data['distance'] == "0":
            dist_str = ""
        else:
            dist_str = f"dl-{halo_data['distance']:.1f}Mpc_"
    else:
        w_str = ""
        if part_data['em_model'] == "annihilation":
            dist_str = f"jfactor-{halo_data['j_factor']:.1e}_"
        else:
            dist_str = f"dfactor-{halo_data['d_factor']:.1e}_"

    if calc_data['freq_mode'] in ['gamma','sgamma','radio','all']:
        fm_str = calc_data['electron_mode']+"_"
    else:
        fm_str = ""
    
    if part_data['em_model'] == "decay":
        wimp_str = "decay_"
    else:
        wimp_str = "annihilation_"

    mx_str = "mx-"
    for mx in calc_data['m_wimp']:
        mx_str += f"{int(mx)}"
        if not mx == calc_data['m_wimp'][-1]:
            mx_str += "-"
        else:
            mx_str += "GeV_"

    diff_str = ""
    if diff_data['loss_only']:
        if calc_data['freq_mode'] in ['gamma','sgamma','radio','all']:
            diff_str = "loss-only_"
    else:
        if calc_data['freq_mode'] in ['gamma','sgamma','radio','all']:
            diff_str = f"D0-{diff_data['diff_constant']}_"

    model_str = part_data['part_model']+"_"
    if not tag is None:
        tag_str = tag+"_"
    else:
        tag_str = ""
    return halo_data['name']+"_"+model_str+mx_str+wimp_str+dm_str+fm_str+w_str+dist_str+diff_str+tag_str

def flux_label(calc_data):
    """
    Find frequency label string for calculation
    
    Arguments
    ---------------------------
    calc_data : dictionary
        Calculation properties

    Returns
    ---------------------------
    flux_str : string
        Frequency label for output file
    """
    if calc_data['freq_mode'] == "radio":
        flux_str = "sync"
    elif calc_data['freq_mode'] == "gamma":
        flux_str = "gamma"
    elif calc_data['freq_mode'] == "pgamma":
        flux_str = "primary_gamma"
    elif calc_data['freq_mode'] == "sgamma":
        flux_str = "secondary_gamma"
    elif "neutrinos" in calc_data['freq_mode']:
        flux_str = calc_data['freq_mode']
    else:
        flux_str = "multi_frequency"
    return flux_str

def process_dict(d,exclude=None,exclude_recurse=False):
    new_d = {}
    if exclude_recurse:
        r_ex = exclude
    else:
        r_ex = None
    for key,value in d.items():
        if isinstance(value,dict):
            new_d[key] = process_dict(value,r_ex)
        else:
            try:
                new_d[key] = np.array(value).tolist()
            except:
                if isinstance(value,np.generic):
                    new_d[key] = value.item()
                else:
                    new_d[key] = value
    if not exclude is None:
        for x in exclude:
            if x in new_d.keys():
                new_d.pop(x)
    return new_d

def make_output(calc_data,halo_data,part_data,mag_data,gas_data,diff_data,cosmo_data,out_mode="yaml",tag=None,em_only=False,no_numpy=False):
    """
    Write calculation data to an output file
    
    Arguments
    ---------------------------
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties
    part_data : dictionary
        Particle physics properties
    mag_data : dictionary
        Magnetic field properties
    gas_data : dictionary
        Gas density properties
    diff_data : dictionary
        Diffusion properties
    cosmo_data : dictionary
        Cosmology properties
    out_mode : string, optional
        Can be "yaml" or "json"
    tag : string, optional
        String added to file name
    em_only: boolean, optional
        True means calc_data['results']['final_data'] is not written
    no_numpy : boolean, optional
        True means all data converted to native python formats, False writes numpy arrays as is (yaml only) 

    Returns
    ---------------------------
    None
    """
    if np.any(calc_data['results']['final_data'] is None):
        fatal_error("output.make_output() cannot be invoked without a full set of calculated results, some masses have not had calculations run")
    def default(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        raise TypeError('Unknown type:', type(obj))

    if no_numpy and out_mode=="yaml":
        write_halo = process_dict(halo_data,exclude=['halo_density_func'])
        write_gas = process_dict(gas_data,exclude=['gas_density_func'])
        write_mag = process_dict(mag_data,exclude=['mag_field_func'])
        write_part = process_dict(part_data,exclude=['d_ndx_interp'])
        write_diff = process_dict(diff_data)
        write_calc = process_dict(calc_data)
        write_cosmo = process_dict(cosmo_data)
    else:
        write_halo = {key: value for key, value in halo_data.items() if not key == 'halo_density_func'}
        write_mag = {key: value for key, value in mag_data.items() if not key == 'mag_field_func'}
        write_gas = {key: value for key, value in gas_data.items() if not key == 'gas_density_func'}
        write_part = {key: value for key, value in part_data.items() if not key == 'd_ndx_interp'}
        write_calc = {key: value for key, value in calc_data.items()}
        write_cosmo = cosmo_data
        write_diff = diff_data

    if em_only and calc_data['calc_mode'] == "jflux":
        warning("output.make_output(): jflux calculations have no emissivity, em_only = True cannot be used!, reverting to em_only = False")
        em_only = False
    if em_only:
        write_calc['results'].pop('final_data')
    out_data = {'calc_data':write_calc,'halo_data':write_halo,'part_data':write_part,'mag_data':write_mag,'gas_data':write_gas,'diff_data':write_diff,'cosmo_data':write_cosmo}
    f_name = get_calc_id(calc_data,halo_data,part_data,diff_data,tag=tag)+flux_label(calc_data)
    if not em_only:
        f_name += "_"+calc_data['calc_mode']
    else:
        f_name += "_emissivity"
    if out_mode == "yaml":
        out_file = open(f_name+".yaml","w")
        yaml.dump(out_data, out_file)
        out_file.close()
    elif out_mode == "json":
        out_file = open(f_name+".json","w")
        json.dump(out_data,out_file,default=default)
        out_file.close()

def wimp_write(mx,part_data,target=None):
    """
    Write WIMP parameters to a target output
    
    Arguments
    ---------------------------
    mx : float
        WIMP mass [GeV]
    part_data : dictionary
        Particle physics properties
    target : string (optional)
        Output file or stream name, None goes to stdout

    Returns
    ---------------------------
    None
    """
    class string_stream:
        def __init__(self,string):
            self.text = string

        def write(self,string):
            self.text += string
    end = "\n"
    if(target is None):
        out_stream = sys.stdout
    elif os.path.isfile(target):
        out_stream = open(target,"w")
    else:
        out_stream = string_stream(target)
        end = ""
    if target is None:
        prefix = ""
    else:
        prefix = "#"
    out_stream.write(f"{prefix}{'='*spacer_length} {end}")
    out_stream.write(f"{prefix}Now calculating for Dark Matter model: {end}")
    out_stream.write(f"{prefix}{'='*spacer_length}{end}")
    out_stream.write(f"{prefix}WIMP mass: {mx} GeV{end}")
    out_stream.write(f"{prefix}Particle physics: {part_data['part_model']}{end}")
    out_stream.write(f"{prefix}Emission type: {part_data['em_model']}{end}")
    if part_data["em_model"] == "annihilation" and "cross_section" in part_data.keys():
        out_stream.write(f"{prefix}Cross-section: {part_data['cross_section']} m^3 s^-1{end}")
    elif part_data["em_model"] == "decay" and "decay_rate" in part_data.keys():
        out_stream.write(f"{prefix}Decay rate: {part_data['decay_rate']} s^-1{end}")

def calc_write(calc_data,halo_data,part_data,mag_data,gas_data,diff_data,target=None):
    """
    Write calculation data to a target output
    
    Arguments
    ---------------------------
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties
    part_data : dictionary
        Particle physics properties
    mag_data : dictionary
        Magnetic field properties
    gas_data : dictionary
        Gas density properties
    diff_data : dictionary
        Diffusion properties
    target : string (optional)
        Output file or stream name, None goes to stdout

    Returns
    ---------------------------
    None
    """
    class string_stream:
        def __init__(self,string):
            self.text = string

        def write(self,string):
            self.text += string
    gas_params = yaml.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"config/gas_density_profiles.yaml"),"r"),Loader=yaml.SafeLoader)
    mag_params = yaml.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"config/mag_field_profiles.yaml"),"r"),Loader=yaml.SafeLoader)
    unit_dict = {"distance":"Mpc","magnetic":"micro-Gauss","num_density":"cm^-3"}
    end = "\n"
    string_out = False
    if(target is None):
        out_stream = sys.stdout
    elif os.path.isfile(target):
        out_stream = open(target,"w")
    else:
        out_stream = string_stream(target)
        end = ""
        string_out = True
    if target is None:
        prefix = ""
    else:
        prefix = "#"
    out_stream.write(f"{prefix}{'='*spacer_length}{end}")
    out_stream.write(f"{prefix}Run Parameters{end}")
    out_stream.write(f"{prefix}{'='*spacer_length}{end}")
    if 'calc_output_directory' in calc_data.keys():
        out_stream.write(f"{prefix}Output directory: {calc_data['calc_output_directory']}{end}")
    #out_stream.write(f"{prefix}Field File Code: b'+str(int(phys.b0))+"q"+str(phys.qb)+end)
    out_stream.write((f"{prefix}Frequency Samples: {calc_data['f_sample_num']}{end}"))
    out_stream.write(f"{prefix}Minimum Frequency Sampled: {calc_data['f_sample_limits'][0]:.2e} MHz ({(calc_data['f_sample_limits'][0]*units.Unit('MHz')*constants.h).to('GeV').value:.2e} GeV) {end}")
    out_stream.write(f"{prefix}Maximum Frequency Sampled: {calc_data['f_sample_limits'][1]:.2e} MHz ({(calc_data['f_sample_limits'][1]*units.Unit('MHz')*constants.h).to('GeV').value:.2e} GeV){end}")
    if not calc_data['calc_mode'] == "jflux":
        out_stream.write(f"{prefix}Radial Grid Intervals: {calc_data['r_sample_num']}{end}")
        if calc_data['freq_mode'] in ['all','radio','sgamma'] and "green" in calc_data['electron_mode']:
            out_stream.write(f"{prefix}Green's Function Grid Intervals: {calc_data['r_green_sample_num']}{end}")
        if diff_data['diff_rmax'] == "2*Rvir":
            r_limit = 2*halo_data['rvir']
        else:
            r_limit = diff_data['diff_rmax']
        out_stream.write(f"{prefix}Minimum Sampled Radius: {halo_data['scale']*10**calc_data['log10_r_sample_min_factor']:.2e} Mpc{end}")
        out_stream.write((f"{prefix}Maximum Sampled Radius: {r_limit:.2e} Mpc{end}"))
    out_stream.write(f"{prefix}{'='*spacer_length}{end}")
    out_stream.write(f"{prefix}Halo Parameters: {end}")
    out_stream.write(f"{prefix}{'='*spacer_length}{end}")
    out_stream.write(f"{prefix}Halo Name: {halo_data['name']}{end}")
    if 'z' in halo_data.keys():
        out_stream.write(f"{prefix}Redshift z: {halo_data['z']:.2e}{end}")
    if 'distance' in halo_data.keys():
        out_stream.write(f"{prefix}Luminosity Distance: {halo_data['distance']:.2e} Mpc{end}")
    if 'profile' in halo_data.keys():
        out_stream.write(f"{prefix}Halo profile: {halo_data['profile']}{end}")
        if halo_data['profile'] in ["einasto","gnfw","cgnfw"]:
            out_stream.write(f"{prefix}Halo index parameter: {halo_data['index']}{end}")
    if 'mvir' in halo_data.keys():
        out_stream.write(f"{prefix}Virial Mass: {halo_data['mvir']:.2e} Solar Masses{end}")
    if 'rvir' in halo_data.keys():
        out_stream.write(f"{prefix}Virial Radius: {halo_data['rvir']:.2e} Mpc{end}")
    if 'scale' in halo_data.keys():
        out_stream.write(f"{prefix}Halo scale radius: {halo_data['scale']:.2e} Mpc{end}")
    if 'rho_norm_relative' in halo_data.keys():
        out_stream.write(f"{prefix}Rho_s/Rho_crit: {halo_data['rho_norm_relative']:.2e}{end}")
    if 'cvir' in halo_data.keys():
        out_stream.write(f"{prefix}Virial Concentration: {halo_data['cvir']:.2f}{end}")
    if calc_data['calc_mode'] == "jflux" and 'truncation_scale' in halo_data.keys():
        out_stream.write(f"{prefix}Truncation/tidal radius: {halo_data['truncation_scale']:.2e} Mpc{end}")
    if part_data['em_model'] == "decay" and calc_data['calc_mode'] == "jflux":
        out_stream.write(f"{prefix}Dfactor: {halo_data['j_factor']:.2e} GeV cm^-2{end}")
    elif part_data['em_model'] == "annihilation" and calc_data['calc_mode'] == "jflux":
        out_stream.write(f"{prefix}Jfactor: {halo_data['j_factor']:.2e} GeV^2 cm^-5{end}")

    if calc_data['freq_mode'] in ["all","sgamma","radio"]:
        out_stream.write(f"{prefix}{'='*spacer_length}{end}")
        out_stream.write(f"{prefix}Gas Parameters: {end}")
        out_stream.write(f"{prefix}{'='*spacer_length}{end}")
        out_stream.write(f"{prefix}Gas density profile: {gas_data['profile']}{end}")
        param_set = gas_params[gas_data['profile']]
        for p in param_set:
            unit_type = check_quant(p)
            if not unit_type is None:
                out_stream.write(f"{prefix}{p}: {gas_data[p]} {unit_dict[unit_type]} {end}")
            else:
                out_stream.write(f"{prefix}{p}: {gas_data[p]}{end}")
        out_stream.write(f"{prefix}{'='*spacer_length}{end}")
        out_stream.write(f"{prefix}Magnetic Field Parameters: {end}")
        out_stream.write(f"{prefix}{'='*spacer_length}{end}")
        out_stream.write(f"{prefix}Magnetic field profile: {mag_data['profile']}{end}")
        param_set = mag_params[mag_data['profile']]
        for p in param_set:
            unit_type = check_quant(p)
            if not unit_type is None:
                out_stream.write(f"{prefix}{p}: {mag_data[p]} {unit_dict[unit_type]} {end}")
            else:
                out_stream.write(f"{prefix}{p}: {mag_data[p]}{end}")
        if diff_data['loss_only']:
            out_stream.write(f"{prefix}No Diffusion{end}")
        else:
            out_stream.write(f"{prefix}Spatial Diffusion{end}")
            out_stream.write(f"{prefix}Turbulence index: {diff_data['diff_index']:.2f}{end}")
            out_stream.write(f"{prefix}Diffusion constant: {diff_data['diff_constant']:.2e} cm^2 s^-1{end}")
    if string_out:
        return out_stream.text
    elif not target is None:
        out_stream.close()

def fits_map(sky_coords,target_freqs,calc_data,halo_data,part_data,diff_data,sigv=1e-26,max_pix=6000,display_slice=None,r_max=None,target_resolution=5.0/60):
    """
    Output a fits file with radio maps
    
    Arguments
    ---------------------------
    sky_coords : SkyCoord (astropy object)
        Sky coordinates object from astropy
    target_freqs : list
        Frequencies for fits slices [MHz]
    calc_data : dictionary
        Calculation properties
    halo_data : dictionary
        Halo properties
    part_data : dictionary
        Particle physics properties
    mag_data : dictionary
        Magnetic field properties
    gas_data : dictionary
        Gas density properties
    diff_data : dictionary
        Diffusion properties
    sigv : float, optional
        Cross-section or decay rate [cm^3 s^-1 or s^-1]
    max_pix : int, optional
        Maximum number of image pixels per axis
    target_resolution : float, optional
        Angular width of pixel (will be final resolution unless it exceeds max_pix limit) [arcmin]
    display_slice : int, optional
        Index of frequency to display in a plot
    r_max : float, optional
        Radial extent of fits map to be saved [Mpc]

    Returns
    ---------------------------
    None
    """
    target_freqs = np.atleast_1d(target_freqs)
    if not calc_data['calc_mode'] == "sb":
        fatal_error("output.fits_map() can only be run with surface brightness data")
    if np.any(calc_data['results']['final_data'] is None):
        fatal_error("output.fits_map() cannot be invoked without a full set of calculated results, some masses have not had calculations run")
    if np.any(target_freqs < calc_data['f_sample_limits'][0]) or np.any(target_freqs > calc_data['f_sample_limits'][-1]):
        fatal_error(f"Requested frequencies lie outside the calculation range {calc_data['f_sample_limits'][0]} - {calc_data['f_sample_limits'][-1]} MHz")
    #we use more pixels than we want to discard outer ones with worse resolution distortion from ogrid

    if not display_slice is None:
        try:
            int(display_slice)
        except:
            fatal_error("output.fits_map() parameter display_slice must be an integer that addresses an element of target_freqs")
        if display_slice >= len(target_freqs) or display_slice < 0 or display_slice != int(display_slice):
            fatal_error("output.fits_map() parameter display_slice must be an integer that addresses an element of target_freqs")
    if diff_data['diff_rmax'] == "2*Rvir":
        r_limit = 2*halo_data['rvir']
    else:
        r_limit = diff_data['diff_rmax']
    if not r_max is None:
        if r_max > r_limit:
            fatal_error(f"output.fits_map() argument r_max cannot be greater than largest sampled r value {r_limit} Mpc")
    else:
        r_max = r_limit
    half_pix = int(np.arctan(r_limit/halo_data['distance'])/np.pi*180*60/target_resolution)
    if half_pix > max_pix/2:
        half_pix = int(max_pix/2)
    use_half_pix = int(half_pix*r_max/r_limit)
    use_start_pix = half_pix-use_half_pix
    use_end_pix = use_start_pix + 2*use_half_pix
    r_set = np.logspace(np.log10(halo_data['scale']*10**calc_data['log10_r_sample_min_factor']),np.log10(r_limit),num=calc_data['r_sample_num'])
    r_set = np.arctan(r_set/halo_data['distance'])/np.pi*180*60 #must be arcmins for the algorithm below
    f_set = calc_data['f_sample_values']
    r_max = np.arctan(r_max/halo_data['distance'])/np.pi*180*60
    hdu_list = []
    for mx in calc_data['m_wimp']:
        fits_out_set = []
        mass_index = np.where(calc_data['m_wimp']==mx)[0][0]
        if len(calc_data['f_sample_values']) == 1:
            full_data_intp = interp1d(r_set,calc_data['results']['final_data'][mass_index][0],bounds_error=False,fill_value=0.0)
        else:
            full_data_intp = RegularGridInterpolator((f_set,r_set),calc_data['results']['final_data'][mass_index],bounds_error=False,fill_value=0.0)#interp2d(r_set,f_set,calc_data['results']['final_data'][mass_index],bounds_error=False,fill_value=0.0)
        for i in range(len(target_freqs)):
            if len(calc_data['f_sample_values']) == 1:
                intp = full_data_intp
            else:
                r_data = full_data_intp((target_freqs[i]*np.ones_like(r_set),r_set))
                intp = interp1d(r_set,r_data)
            circle = np.ogrid[-half_pix:half_pix,-half_pix:half_pix]
            r_plot = np.sqrt(circle[0]**2  + circle[1]**2)
            n = circle[0].shape[0]
            angle_alpha = (r_set[-1]-r_set[0])/(n-1) #for a conversion from array index to angular values
            angle_beta = r_set[-1] - angle_alpha*(n-1)
            r_plot = angle_alpha*r_plot + angle_beta #linear set of angular samples (1 per pixel) - upgrade to binned?
            arcmin_per_pixel = r_set[-1]*2/n
            s_plot = intp(r_plot*1.00000001)*arcmin_per_pixel**2
            if part_data['em_model'] == "annihilation":
                s_plot *= sigv/part_data['cross_section']
            else:
                s_plot *= sigv/part_data['decay_rate']
            ra_val = sky_coords.ra.value*60 #arcmin
            dec_val = sky_coords.dec.value*60 #arcmin
            if not display_slice is None:
                if i == display_slice:
                    fig = plt.gcf()
                    ax = fig.gca()
                    ax.set_aspect('equal')
                    im = plt.imshow(s_plot[use_start_pix:use_end_pix,use_start_pix:use_end_pix],cmap="RdYlBu",norm=LogNorm(vmin=np.min(s_plot[use_start_pix:use_end_pix,use_start_pix:use_end_pix]),vmax=np.max(s_plot[use_start_pix:use_end_pix,use_start_pix:use_end_pix])),extent=[(r_max+ra_val)/60,(-r_max+ra_val)/60,(-r_max+dec_val)/60,(r_max+dec_val)/60])
                    plt.xlabel("RA--SIN (degrees)")
                    plt.ylabel("DEC--SIN (degrees)")
                    cbar = plt.colorbar(im)
                    cbar.set_label(r"I$(\nu)$ Jy/pixel")
                    plt.show()
            fits_out_set.append(s_plot[use_start_pix:use_end_pix,use_start_pix:use_end_pix])
        fits_out_set = np.array(fits_out_set)

        angle_alpha = 2*r_set[-1]/(n-1)
        angle_beta = r_set[-1] - angle_alpha*(n-1)

        ra_set = np.flipud((np.arange(n)*angle_alpha+angle_beta+ra_val)/60) #ra declines to the right in RA---SIN
        dec_set = (np.arange(n)*angle_alpha+angle_beta+dec_val)/60 

        ra_delta = np.sum(-ra_set[0:-1]+ra_set[1:])/(n-1)
        dec_delta = np.sum(-dec_set[0:-1]+dec_set[1:])/(n-1)

        #creating a world coordinate system for our fits file
        w_coords = wcs.WCS(naxis=2)
        w_coords.wcs.crpix = [use_half_pix,use_half_pix]
        w_coords.wcs.crval = [ra_set[use_start_pix+use_half_pix],dec_set[use_start_pix+use_half_pix]]
        w_coords.wcs.ctype = ['RA---SIN','DEC--SIN']
        w_coords.wcs.cdelt = [ra_delta,dec_delta]

        hdr = w_coords.to_header()
        if mass_index == 0:
            hdu = fits.PrimaryHDU(fits_out_set,header=hdr)
        else:
            hdu = fits.ImageHDU(fits_out_set,header=hdr)
        hdr = hdu.header
        hdr['BUNIT'] = 'JY/PIXEL'
        hdr['BZERO'] = 0.0
        hdr['BSCALE'] = 1.0
        hdr['EQUINOX'] = 2000
        hdr['BTYPE'] = 'INTENSITY'
        hdr['ORIGIN'] = 'DARKMATTERS'
        hdr['OBSERVER'] = 'DARKMATTERS'
        hdr['OBJECT'] = halo_data['name'].upper()
        hdr['CTYPE3'] = 'FREQ'
        hdr['CRPIX3'] = 1
        hdr['CRVAL3'] = target_freqs[0]*1e6
        if len(target_freqs) == 1:
            hdr['CDELT3'] = 0
        else:
            hdr['CDELT3'] = (target_freqs[1] - target_freqs[0])*1e6
        hdr['CUNIT3'] = 'HZ'
        hdr['CTYPE4'] = 'STOKES'
        hdr['CRPIX4'] = 1
        hdr['CRVAL4'] = 1
        hdr['CDELT4'] = 1
        hdr['CUNIT4'] = ''
        hdr['WIMPMASS'] = f'{mx}'
        hdr['MASSUNIT'] = "GEV"
        hdu_list.append(hdu)

    hdu_list = fits.HDUList(hdu_list)
    hdu_list.writeto(get_calc_id(calc_data,halo_data,part_data,diff_data)+flux_label(calc_data)+".fits",overwrite=True)
