#cython: language_level=3
from matplotlib.pyplot import csd
import numpy as np
from scipy.integrate import simps as integrate
from scipy import interpolate as sp
try:
    from wimp_tools import cosmology,astrophysics,tools,substructure,ucmh
except:
    import wimp_tools.cosmology as cosmology
    import wimp_tools.tools as tools
    import wimp_tools.environments as environments
    import wimp_tools.astrophysics as astrophysics
    import wimp_tools.substructure as substructure
    import wimp_tools.ucmh as ucmh
try:
    from emm_tools import electron
except:
    import emm_tools.electron as electron
from os.path import isdir,join
from os import mkdir
import sys

class simulation_env:
    """
    A container for parameters related to execution details of calculations
        ---------------------------
        Attributes
        ---------------------------
        n                 - number of samples in the radial direction for majority of uses (int)
        ngr               - radial samples when solving Green's functions (int)
        radio_boost_flag  - use or don't use modified substructure boost for synchrotron (1 or 0) (int)
        flim              - two element list with minimum and maximum fluxes [MHz] (float)
        electrons_from_c  -  True -> use c for electon calculation, False -> use python (boolean)
        exec_electron_c   - c executable file path for electon calculations (String)
        radio_emm_from_c  - True -> use c for radio calculations, False -> use python (boolean)
        exec_emm_c        - c executable file path for radio calculations (String)
        num               - number of flux sampling points for output (int)
        theta             - angular radius for flux integration [arcminutes] (float)
        e_bins            - number of bins in spectra variables in physical_env
        mx_set            - list of WIMP masses to run each calculation with [GeV] (float)
        nu_flavour        - neutrino flavour for neutrino fluxes (String)
        nu_sb             - frequency for surface brightness calculations [MHz] (float)
        sub_mode          - substructure calculation mode (String)
        specdir           - location of WIMP dN/dE files (String)
        rintegrate        - integration radius for flux 
        log_mode          - 1 produce log files, 0 don't (int)
        out_dir           - folder path for output files (String)
        jnormed           - True halo profile normalised to jfactor, False not so (String)
        f_sample          - frequency points for output of fluxes [MHz] (array-like float)
        ---------------------------
        Methods
        ---------------------------
        sample_f   - args : self - generate f_sample from  flim and assign to self
        check_self - args : self - check complete information has been given 

    """
    def __init__(self,n=100,ngr=300,num=40,fmin=1.0,fmax=1e5,theta=1.0,nu_sb=1.4e3):
        self.n = n #radial grid samples for halo data 
        self.ngr = ngr #radial grid samples when solving green's functions, should be larger than n
        self.radio_boost_flag = 0 #this enables subtsructure boost factor reduction based on magnetic field profile
        self.flim = [fmin,fmax] #frequency limits for calculations in MHz
        self.electrons_from_c = False #flag for using c code for electron calculations
        self.exec_electron_c = None #where the c executable is stored
        self.radio_emm_from_c = False #don't use this, isn't working
        self.exec_emm_c = None #as above
        self.num = num #number of frequency samples between fmin and fmax
        self.theta = theta #the angular radius for flux integration in arcminutes
        self.e_bins = None #number of bins to interpolate dN/dE spectra
        self.mx_set = [10.0,2e1,4e1,7e1,1e2,2e2,3e2,5e2,7e2,1e3,1.3e3,1.5e3,2e3,2.5e3,3e3,5e3,1e4] #default wimp mass set for batch runs
        self.nu_flavour = "mu" #neutrino  flavour
        self.nu_sb = nu_sb #frequency for surface brightness computation in MHz
        self.sub_mode = None #None is no substructure, sc2006 uses sub_frac calculation, prada uses sanchez-conde & prada 2013
        self.specdir = join(sys.path[0],"particle_physics") #directory of input spectra 
        self.rintegrate = None #radius for flux integration in Mpc
        self.log_mode = 0 #do you want log files? 
        self.out_dir = "./" #output file location
        self.jnormed = False
        self.f_sample = None
        self.jNormRatio = None
        self.f_spacing = "log"

    def sample_f(self):
        if not self.f_sample is None:
            self.flim = [min(self.f_sample),max(self.f_sample)]
            self.num = len(self.f_sample)
        elif self.num > 1: #check if we want to do one frequency point or multiple
            if "lin" in self.f_spacing:
                spacerFunc = np.linspace
                fMin = self.flim[0];fMax = self.flim[1]
            else:
                spacerFunc = np.logspace
                fMin = np.log10(self.flim[0]);fMax = np.log10(self.flim[1])
            self.f_sample = spacerFunc(fMin,fMax,num=self.num)
        else:
            self.f_sample = np.array([self.nu_sb])

    def check_self(self):
        #we set everything to lower case for ease of checking
        self.nu_flavour = self.nu_flavour.lower()
        specCheck = isdir(self.specdir) #check the directory exists
        if not specCheck:
            print("Error: could not find directory "+self.specdir)
        if self.nu_flavour in ["e","el","ee","electron"]:
            self.nu_flavour = "e"
        if self.nu_flavour in ["m","mm","muon","mumu"]:
            self.nu_flavour = "mu"
        if self.nu_flavour in ["t","tt","tautau","tauon"]:
            self.nu_flavour = "tau"
        nuCheck = self.nu_flavour in ["mu","e","tau"] #check the nuetrino flavour is sensible
        outCheck = isdir(self.out_dir) #check the output directory exists, if not try make it
        if not outCheck:
            print("Warning: no output directory called "+self.out_dir+" trying to create it")
            try:
                mkdir(self.out_dir)
                outCheck = True
                print("Created directory: "+self.out_dir)
            except:
                print("Error: could not create directory "+self.out_dir)
        if not nuCheck:
            print("Warning: nu_flavour must be one of [mu,e,tau]")
        if self.sub_mode is None:
            self.sub_mode = "none" #if no substructure mode was specified
        subCheck = self.sub_mode in ["sc2006","prada","none"] #check substructure mode is valid
        if not subCheck:
            print("Warning: submode must be one of [sc2006,prada,none]")
        flimCheck = self.flim[0] <= self.flim[1] and len(self.flim) == 2 #check the frequency limits make sense
        fSampleCheck = not self.f_sample is None
        if flimCheck or fSampleCheck:
            self.sample_f() #make a frequency sample
        else:
            print("Warning: flim must be specified by fmin and fmax, you gave: "+str(self.flim))

        print(self.f_sample)
        if (flimCheck or fSampleCheck) and subCheck and nuCheck and specCheck and outCheck: #must pass all checks to be a valid sim env
            return True
        else:
            return False

    def simFromHeader(self,hdr):
        #need: boost, radio_boost
        r_sample = np.array(hdr['CRSET3'].split(),dtype=np.float64)
        self.n = len(r_sample)
        self.f_sample = np.array(hdr['CRSET2'].split(),dtype=np.float64)
        self.num = len(self.f_sample)
        
class physical_env:
    """
    A container for parameters related to physical environment and particle physics details
        ---------------------------
        Attributes
        ---------------------------
        b0             - magnetic field normalisation [uG] (float)
        qb             - magnetic field model parameter [varies] (float)
        b_flag         - magnetic field model label (String)
        ne_model       - gas density model label (String)
        ne0            - gas density nomalisation [cm^-3] (float)
        lb             - scale radius for baryon density [Mpc] (float)
        qe             - gas density model parameter [varies] (float)
        mx             - WIMP mass [GeV] (float)
        lc             - magnetic field coherence length [kpc] (float)
        delta          - turbulence spectral index [] (float)
        diff           - diffusion flag (1,0) (int)
        channel        - list of annihilation channels to be used (String)
        branching      - list of branching ratios (float)
        spectrum       - two element list containing [g,dN/dg] for positrons, g is Lorentz gamma (E/me)
        gamma_spectrum - two element list containing [g,dN/dg] for photons, g is Lorentz gamma (E/me)
        nu_spectrum    - two element list containing [g,dN/dg] for neutrinos, g is Lorentz gamma (E/me)
        spec_given     - (boolean)
        me             - electron mass-energy [GeV] (float)
        lp             - old variable (not sure if still used)
        btag           - a label for b models (only appears in output) (String)
        specMin        - minimum Lorentz gamma for positrons in spectrum[0] (float)
        specMax        - maximum Lorentz gamma for positrons in spectrum[0] (float)
        gamma_specMin  - minimum Lorentz gammafor photons in gamma_spectrum[0] (float)
        gamma_specMax  - maximum Lorentz gamma for photons in gamma_spectrum[0] (float)
        nu_specMin     - minimum Lorentz gamma for neutrinos in nu_spectrum[0] (float)
        nu_specMax     - maximum Lorentz gamma for neutrinos in nu_spectrum[0] (float)
        particle_model - label for WIMP annihilation/decay model (String)
        all_channels   - list of all channels that there is data for, used for checking input (String)
        d0             - diffusion constant [cm^2 s^-1] (float)
        ISRF           - flag 1 or 0 for using inter-stellar radiation field for inverse-Compton scattering emissions (int)
        ---------------------------
        Methods
        ---------------------------
        clear_spectra   - args : self - empty spectrum,gamma_spectrum,nu_spectrum variables
        check_self      - args : self - check complete information has been given 
        check_particles - args : self - check channels and branching ratios
    """
    def __init__(self,ne_model="flat",b_flag="flat",mx=0.0,channel=["bb"],branching=[1.0],particle_model=None,b0=0.0,qb=0.0,ne0=0.0,qe=0.0,lc=0.0,delta=1.6666,diff=0):
        self.b0 = b0 #magnetic field normalisation
        self.qb = qb #magnetic field paramater, function varies by model
        self.b_flag = b_flag #this records what magnetic field model is in use
        self.ne_model = ne_model #the gas density model
        self.ne0 = ne0 #gas density normalisation
        self.lb = None #scale radius for baryon density
        self.qe = qe #gas density power-law index
        self.mx = mx #WIMP mass
        self.lc = lc #magnetic field coherence length
        self.delta = delta #magnetic field turbulence spectral index
        self.diff = diff #diffusion flag
        self.channel = channel #annihilation channel, is a list (even if only 1 element)
        self.branching = branching #a branching ratio for each channel entry
        self.spectrum = [None,None] #E,dN/dE injection spectra for positrons 
        self.gamma_spectrum = [None,None] #E,dN/dE injection spectra for photons
        self.nu_spectrum = [None,None] #E,dN/dE injection spectra for neutrinos
        self.spec_given = False #not sure - CHECK
        self.me = 0.511e-3 #electron mass GeV
        self.lp = None #not sure if still used
        self.btag = None #a label for b model if you want to specify
        self.specMin = None #min E for spectrum[0]
        self.specMax = None #max E for spectrum[0]
        self.gamma_specMin = None #min E for gamma_spectrum[0]
        self.gamma_specMax = None #max E for gamma_spectrum[0]
        self.nu_specMin = None #min E for nu_spectrum[0]
        self.nu_specMax = None #max E for nu_spectrum[0]
        self.particle_model = particle_model #a label you can specify or will be found from channels and branching sets
        self.particle_reset_flag = False #flag telling code to reconstruct particle model with new data
        self.all_channels = ["bb","tautau","qq","mumu","ee","gamma","zz","ww","hh","tt"] #all channel set to check given inputs
        self.d0 = None
        self.ISRF = 0
        self.model_independent = True
        
    def clear_spectra(self): #make sure the spectral data is reset if we are changing particle models
        self.spectrum = [None,None]
        self.gamma_spectrum = [None,None]
        self.nu_spectrum = [None,None]
        
    def check_self(self):
        #simple checks that vital parameters are given
        self.ne_model = self.ne_model.lower() 
        self.b_flag = self.b_flag.lower()
        if(self.b0 != 0.0 and self.ne0 != 0.0 and ((self.diff == 0) or (self.lc != 0.0 and self.diff == 1)) and self.mx != 0.0):
            check = True
        else:
            check = False
        neCheck = self.ne_model in astrophysics.ne_model_set #see if the model is in the set we have formulae for
        if not neCheck:
            print("Warning: ne_model must be one of "+str(astrophysics.ne_model_set))
        bCheck = self.b_flag in astrophysics.b_model_set #see if the model is in the set we have formulae for
        if not bCheck:
            print("Warning: b_model must be one of "+str(astrophysics.b_model_set))
        p_check = self.check_particles()
        if self.mx == 0.0:
            print("Please specify the WIMP mass")
        if self.b0 == 0.0:
            print("Please specify the magnetic field strength normalisation B")
        if self.ne0 == 0.0:
            print("Please specify the plasma density normalisation ne")
        if self.lc == 0.0 and self.diff == 1:
            print("Please specify the turbulence scale d in kpc")
        return check and p_check and neCheck and bCheck
    
    def check_particles(self):
        check = True
        for i in range(0,len(self.channel)):
            self.channel[i] = self.channel[i].lower() #set all to lower to ease comparison
        for ch in self.channel:
            if not ch in self.all_channels:
                check = False
                print("Please specify the particle channels from ",self.all_channels)
                print("channels given: ",self.channel)
        if str(self.particle_model).lower() == "none":
            self.particle_model = None
        if self.particle_model is None or self.particle_reset_flag: #either there is no model or model must be changed
            self.particle_reset_flag = True
            if len(self.channel) == 1: #for simple model independent work
                self.particle_model = self.channel[0]
                self.branching = [1.0]
            else:
                self.particle_model = ""
                if len(self.branching) != len(self.channel):
                    print("Warning: number of branching ratios doesn't match channels, setting all equal!")
                    self.branching = np.ones(len(self.channel),dtype=float)/len(self.channel)
                for ch,br in zip(self.channel,self.branching): #build a string of channels and branching ratios
                    print(("%3.2f"%br)[-1])
                    if float(("%3.2f"%br)[-1]) == 0.0: 
                        self.particle_model += "%2.1f"%br+str(ch)
                    else:
                        self.particle_model += "%3.2f"%br+str(ch)
        elif len(self.branching) != len(self.channel):
            print("Warning: number of branching ratios doesn't match channels, setting all equal!")
            self.branching = np.ones(len(self.channel),dtype=float)/len(self.channel)
            for ch,br in zip(self.channel,self.branching):
                    self.particle_model += "%3.2f"%br+str(ch)
        return check

    def physFromHeader(self,hdr):
        self.particle_model = hdr['PMODEL']
    
    

class halo_env:
    """
    Container for all halo related parameters
        ---------------------------
        Attributes
        ---------------------------
        name                 - halo name for labelling output (String)
        ready                - flag for showing all information is specified for running calculations (boolean)
        electrons            - electron equilibrium distribution storage (2D array phys.e_bins x sim.n floats)
        ucmh                 - 1 or 0, flag to indicate if halo is a ucmh (int)
        phase                - formation epoch for ucmh halo (String)
        sub_frac             - fraction of halo mass found in sub-halos, used in sc2006 substructure model only (float)
        profile              - DM density profile (String)
        dm                   - flag determined by density profile (int)
        alpha                - Einasto parameter (float)
        gnfw_gamma           - density exponent for generalised NFW profile (float)
        J                    - jfactor
        J_flag               - has j-factor been set via input or not (1 or 0) (int)
        tag                  - a labelling variable used for output names if J_flag was set to 1
        mvir                 - halo virial mass [Msol] (float)
        cvir                 - virial concentration [] (float)
        cvir_flag            - was cvir set by input or not (1 or 0) (int)
        rvir                 - virial radius [Mpc] (float)
        rvir_flag            - was rvir set by input or not (1 or 0) (int)
        rcore                - halo scale radius [Mpc] (float)
        rcore_flag           - was rcore set by input or not (1 or 0) (int)
        rhos                 - halo characteristic density relative to critical value [] (float)
        rhos_flag            - was rhos set by input or not (1 or 0) (int)
        rho0                 - halo characteristic density [Msol Mpc^-3] (float)
        rho0_flag            - was rho0 set by input or not (1 or 0) (int)
        r_stellar_half_light - stellar half-light radius [Mpc] (float)
        t_sf                 - star formation time [s] (float)
        z                    - halo redshift (float)
        da                   - angular diamater distance to halo [Mpc] (float)
        dl                   - luminosity distance to halo [Mpc]  (float)
        dl_flag              - was dl set by input or not (1 or 0) (int); 1 makes dl and z independent
        r_sample             - two element list [radial sampling points, radial sampling points for Green's functions] [Mpc,Mpc] (float)
        rho_dm_sample        - two element list [rho_DM^mode_exp at radial sampling points, rho_DM^mode_exp at radial sampling points for Green's functions] [(Msol Mpc^-3)^mode_exp,(Msol Mpc^-3)^mode_exp] (float)
        b_sample             - magnetic field strength at radial sampling points [uG] (float)
        ne_sample            - gas density at radial sampling points [cm^3] (float)
        rfarc                - radius corresponding to 1 arcminute angular radius [Mpc] (float)
        bav                  - average magnetic field strength [uG] (float)
        neav                 - average gas density [cm^-3] (float)
        bav_flag             - was bav set by input or not (1 or 0) (int)
        neav_flag            - was neav set by input or not (1 or 0) (int)
        radio_emm            - radio emmissivity [cm^-3 s^-1] (float sim.num x sim.n)
        gamma_emm            - gamma-ray emmissivity [cm^-3 s^-1] (float sim.num x sim.n)
        he_emm               - gamma-ray and radio emmissivity [cm^-3 s^-1] (float sim.num x sim.n)
        nu_emm               - neutrino emmissivity [cm^-3 s^-1] (float sim.num x sim.n)
        radio_emm_nu         - radio emmissivity at single frequency (float sim.n)
        gamma_emm_nu         - gamma-ray emmissivity at single frequency (float sim.n)
        he_emm_nu            - X-ray and gamma emmissivity at single frequency (float sim.n)
        radio_arcflux        - radio flux in limited region (sim.theta or sim.rintegrate) [Jy] (float sim.num)
        radio_virflux        - radio flux within rvir [Jy] (float sim.num)
        he_arcflux           - high-energy flux in limited region (sim.theta or sim.rintegrate) [Jy] (float sim.num)
        he_virflux           - high-energy flux within rvir [Jy] (float sim.num)
        multi_arcflux        - multi-frequency flux in limited region (sim.theta or sim.rintegrate) [Jy] (float sim.num) 
        multi_virflux        - multi-frequency flux within rvir [Jy] (float sim.num)
        nu_arcflux           - neutrino flux in limited region (sim.theta or sim.rintegrate) [Jy] (float sim.num)
        nu_virflux           - neutrino flux within rvir [Jy] (float sim.num)
        rb1                  - angular radius 1 for CPU2006 Coma paper [arcminutes] (float) 
        rb2                  - angular radius 2 for CPU2006 Coma paper [arcminutes] (float)
        rb1Dist              - rb1 converted to a distance [Mpc] (float)
        rb2Dist              - rb2 converted to a distance [Mpc] (float)
        boost                - substuctre boosting factor for flux (float)
        radio_boost          - substucture boosting factor for synchrotron flux (float)
        mode                 - specifies if wimp is annihilating or decaying (String)
        mode_exp             - set from mode -  2.0 for annihilation, 1.0 for decay (float)
        ---------------------------
        Methods
        ---------------------------
        check_self           - args : self - input data consistency check 
        make_spectrum        - args : self - build multi-frequency flux spectra
        physical averages    - args : self, rmax(float) - find bav and neav with integration region limited to rmax
        setup                - args : self, sim, phys, cosmo - setup halo values from input data
        setup_ucmh           - args : self, sim, phys, cosmo - setup ucmh halo values from input data
        setup_halo           - args : self, sim, phys, cosmo - setup non-ucmh halo values from input data

    """
    def __init__(self,z=None,m=None,fs=0.0,dmmod=None,alpha=None,name=None,mode="ann"):
        self.name = name #just halo name for labelling output files
        self.ready = False #if halo has been set-up or not
        self.electrons = None #annihilation product electron density
        #===================================================
        # Halo Type Parameters
        #===================================================
        self.ucmh = 0 #0 -> normal halo, 1 -> ultra-compact halo created in a phase transition 
        self.phase = 'EE' #ucmh collapse phase transition defaults to E^+E^- annihilation
        #===================================================
        # Density Profile Parameters
        #===================================================
        self.sub_frac = fs #fraction of halo in sub-halos for Colafrancesco 2006 model
        self.profile = "nfw" #density profile
        self.dm = dmmod #flag to decide halo density profile, -1 -> Einasto, 2 -> Burkert, 1 - > NFW, 3 -> Isothermal
        self.alpha = alpha #Einasto parameter
        self.gnfw_gamma = 1.0 #generalised nfw index
        self.profileCalcMode = 0 #how we compute the densty profile
        #===================================================
        # Jfactor Parameters
        #===================================================
        self.J = None #jfactor
        self.J_flag = 0 #calculate gamma-flux from J-factor rather than full halo formalism
        self.tag = None #label used if we insert a jfactor
        self.Dfactor = None
        #===================================================
        # Virial Halo Parameters
        #===================================================
        self.mvir = m #virial mass in Msol
        self.cvir = None #virial concentration
        self.cvir_flag = 0 #1 -> cvir is from it not from cvir(Mvir)
        self.rvir = None #virial radius in Mpc
        self.rvir_flag = 0 #has rvir been set manually
        self.rcore = None #halo characteristic radius in Mpc, rcore = rvir/cvir
        self.rcore_flag = 0 # 1 -> rcore is from it, 0 -> rcore is calculated as above
        self.rhos = None #characteristic density to normalise halo profile to Mvir (units of rho_crit)
        self.rhos_flag = 0 # 1 -> rhos from it, 0 -> rhos calculated from Mvir
        self.rho0 = None #unitful characteristic density in Msol Mpc^-3
        self.rho0_flag = 0 #1 -> rho0 was set manually
        self.r_stellar_half_light = None #stellar half-light radius in Mpc
        self.t_sf = None #star formation time in s
        #===================================================
        # Cosmic Distances
        #===================================================
        self.z = z #redshift
        self.da = None #angular diameter distance
        self.dl = None #luminosity distance
        self.dl_flag = 0 # 1 -> dl set manually, 0 -> calculate from z
        #===================================================
        # Sampling points for halo properties
        #===================================================
        self.r_sample = [None,None]  #radial samples, 0 is normal use, 1 is for Green's functions
        self.rho_dm_sample = [None,None] ##radial samples of rho_dm, 0 is normal use, 1 is for Green's functions
        self.b_sample = None #magnetic field samples, call the cosmology.bfield function after initialising the halo
        self.ne_sample = None #gas density radial samples
        self.rfarc = None #radius within corresponding to 1 arcminute view
        #===================================================
        # Average magnetic field and thermal electron densities
        #===================================================
        self.bav = 0.0 #average magnetic field strength, if not set is found within rvir
        self.neav = 0.0 #average gas density, if not set is found within rvir
        self.bav_flag = 0
        self.neav_flag = 0
        #===================================================
        # Emissivity data
        #===================================================
        self.radio_emm = None #radio emm
        self.gamma_emm = None #gamma emm
        self.he_emm = None #brem + ics emm
        self.nu_emm = None #neutrino emmissivity   
        #===================================================
        # Fluxes
        #===================================================
        self.radio_arcflux = [] #radio flux within sim.theta arcminutes or sim.r_integrate
        self.he_arcflux = [] #high energy flux in sim.theta arcmin or sim.r_integrate
        self.multi_arcflux = [] #total flux in sim.theta arcmin or sim.r_integrate
        self.radio_virflux = [] #radio flux in virial radius
        self.he_virflux = [] #high energy flux in virial radius
        self.multi_virflux = [] #total virial radius flux
        self.nu_virflux = [] #neutrino flux in virial radius
        self.nu_arcflux = [] #neutrino flux in sim.theta or sim.r_integrate
        #===================================================
        # Angular radii for Colafrancesco 2007 Coma paper
        #===================================================
        self.rb1 = None 
        self.rb2 = None 
        self.rb1Dist = None #converted to distance
        self.rb2Dist = None 
        #===================================================
        # Boost Factors from Substructure
        #===================================================
        self.boost = 1.0
        self.radio_boost = 1.0
        #===================================================
        # Emission Mode Parameters 
        #===================================================
        self.mode = mode.lower()
        self.mode_exp = 2.0 #2.0 -> annihilation, 1.0 -> decay
        self.weights = None
        self.profileDict = {"einasto":-1,"nfw":1,"burkert":2,"isothermal":3}
    
    def check_self(self):
        if self.ucmh != 0:
            check = True
        else:
            haloCheck = True
            if "ann" in self.mode.lower(): #we first setup the WIMP -> SM mode
                self.mode = "ann"
                self.mode_exp = 2.0
            elif self.mode.lower() == "decay":
                self.mode = "decay"
                self.mode_exp = 1.0
            else:
                self.mode = "ann"
                self.mode_exp = 2.0
            zNone = (not self.z is None) or (not self.dl is None) #only actually need one of these
            if not zNone:
                print("Warning: please specify either a redshift z or distance dist for the halo")
            elif self.z is None:
                self.z = 0.0
            if self.profile.lower() == "nfw":
                 self.profile = "nfw"
                 self.dm = 1
            elif(self.profile.lower() == "einasto" or self.profile.lower() == "ein"):
                self.profile = "einasto"
                self.dm = -1
                if self.alpha is None:
                    print("Warning: Einasto alpha parameter not set, defaulting to 0.17")
                    self.alpha = 0.17
            elif(self.profile.lower() == "moore"):
                self.profile = "gnfw"
                self.dm = 1.5
                self.gnfw_gamma = 1.5
            elif(self.profile.lower() == "gnfw"):
                self.profile = "gnfw"
                self.dm = self.gnfw_gamma
            elif(self.profile.lower() == "burkert"):
                self.profile = "burkert"
                self.dm = 2
            elif(self.profile.lower() == "isothermal" or "iso" in self.profile.lower()):
                self.profile = "isothermal"
                self.dm = 3
            else:
                print("Warning: halo density profile: "+str(self.profile)+" not recognised\nOptions are burkert,nfw,gnfw,moore,einasto,isothermal")
                haloCheck = False

            #this is to check we have enough halo information, experimental
            rvirInfo = not (self.mvir is None and self.rvir is None)
            rhoInfo = not (self.rhos is None and self.rho0 is None)
            rsInfo = not (self.rcore is None)
            cvirInfo = not self.cvir is None 
            if rsInfo and rhoInfo:
                haloInfoCheck = True
                self.profileCalcMode = 0
            elif rvirInfo and rsInfo:
                haloInfoCheck = True
                self.profileCalcMode = 1
            elif rvirInfo and cvirInfo:
                haloInfoCheck = True
                self.profileCalcMode = 2
            else:
                haloInfoCheck = False
            if zNone and haloInfoCheck and haloCheck:
                check = True
            else:
                check = False
        return check

    def make_spectrum(self):
        #builds the total photon output spectrum from the radio and high energy components
        if self.radio_arcflux != [] and self.he_arcflux != []:
           self.multi_arcflux = self.radio_arcflux + self.he_arcflux 
        if self.radio_virflux != [] and self.he_virflux != []:
           self.multi_virflux = self.radio_virflux + self.he_virflux 

    def physical_averages(self,rmax):
        if self.weights is None or self.weights == "rho":
            weights = self.rho_dm_sample[0] #the average is weighted
        elif self.weights == "flat":
            weights = np.ones(len(self.rho_dm_sample[0]),dtype=np.float64)
        if self.bav_flag == 0:
            self.bav = tools.weightedVolAvg(self.b_sample,self.r_sample[0],weights,rmax)
        if self.neav_flag == 0:
            self.neav = tools.weightedVolAvg(self.ne_sample,self.r_sample[0],weights,rmax)

    def reset_avgs(self):
        self.bav_flag = 0
        self.bav = 0.0
        self.neav_flag = 0
        self.neav = 0.0

    def rhoNorm(self,cosmo):
        if self.rho0 is None and not self.rhos is None:
            self.rho0 = self.rhos*cosmology.rho_crit(self.z,cosmo)
        elif self.rhos is None and not self.rho0 is None:
            self.rhos = self.rho0/cosmology.rho_crit(self.z,cosmo)
        else:
            self.rho0 = self.mvir/astrophysics.rho_volume_int(self.rvir,self.rcore,1,self.dm,self.alpha)
            self.rhos = self.rho0/cosmology.rho_crit(self.z,cosmo)

    def rhoProfileCalc(self,cosmo):
        if self.profileCalcMode == 1:
            if not self.rvir is None:
                if self.cvir is None:
                    self.cvir = self.rvir/self.rcore
                if self.mvir is None:
                    self.mvir = cosmology.mvir_from_rvir(self.rvir,self.z,cosmo)
                self.rhoNorm(cosmo)
            else:
                if self.rvir is None:
                    self.rvir = cosmology.rvir(self.mvir,self.z,cosmo)
                if self.cvir is None:
                    self.cvir = self.rvir/self.rcore
                self.rhoNorm(cosmo)
        elif self.profileCalcMode == 2:
            if not self.rvir is None:
                if self.rcore is None:
                    self.rcore = self.rvir/self.cvir
                if self.mvir is None:
                    self.mvir = cosmology.mvir_from_rvir(self.rvir,self.z,cosmo)
                self.rhoNorm(cosmo)
            else:
                if self.rvir is None:
                    self.rvir = cosmology.rvir(self.mvir,self.z,cosmo)
                if self.rcore is None:
                    self.rcore = self.rvir*self.cvir
                self.rhoNorm(cosmo)
        else:
            if self.rhos is None:
                self.rhos = self.rho0/cosmology.rho_crit(self.z,cosmo)
            elif self.rho0 is None:
                self.rho0 = self.rhos*cosmology.rho_crit(self.z,cosmo)
            if self.mvir is None and not self.rvir is None:
                self.mvir = self.rho0*astrophysics.rho_volume_int(self.rvir,self.rcore,1,self.dm,self.alpha)
            if self.rvir is None and not self.mvir is None:
                self.rvir = cosmology.rvir(self.mvir,self.z,cosmo)
            else:
                self.rvir = astrophysics.rvir_from_rho(self.z,self.rhos,self.rcore,self.dm,cosmo,self.alpha)
                self.mvir = self.rho0*astrophysics.rho_volume_int(self.rvir,self.rcore,1,self.dm,self.alpha)
            if self.cvir is None and not self.rvir is None:
                self.cvir = self.rvir/self.rcore
    def setup(self,sim,phys,cosmo):
        #vital to force re-calculations if the halo is changed
        self.radio_emm = None
        self.gamma_emm = None
        self.he_emm = None
        self.radio_emm_nu = None
        self.electrons = None
        #decide what kind of halo  to set up
        if self.ucmh == 0:
            self.setup_halo(sim,phys,cosmo)
        else:
            self.setup_ucmh(sim,phys,cosmo)
        return self.ready

    def setup_ucmh(self,sim,phys,cosmo):
        #sets up an ultra-compact halo instead
        if self.check_self() and sim.check_self() and phys.check_self():# and not sim.specdir is None:
            radians_per_arcmin = 2.909e-4
            #self.mvir = ucmh.massUCMH(self.z,self.phase,**cosmo.cosmo)
            self.rvir = ucmh.rConvert(self.z,self.mvir,**cosmo.cosmo) 
            self.r_sample = [np.logspace(np.log10(self.rvir*1e-7),np.log10(2*self.rvir),sim.n),np.logspace(np.log10(self.rvir*1e-7),np.log10(2*self.rvir),sim.ngr)]
            if self.dm == 1.5:
                self.rho_dm_sample = [ucmh.rhoUCMHMoore(self.r_sample[0],self.z,self.mvir,1.0e-26,phys.mx,**cosmo.cosmo)**self.mode_exp,ucmh.rhoUCMHMoore(self.r_sample[1],self.z,self.mvir,1.0e-26,phys.mx,**cosmo.cosmo)**self.mode_exp]
            else:
                self.rho_dm_sample = [ucmh.rhoUCMH(self.r_sample[0],self.z,self.mvir,1.0e-26,phys.mx,**cosmo.cosmo)**self.mode_exp,ucmh.rhoUCMH(self.r_sample[1],self.z,self.mvir,1.0e-26,phys.mx,**cosmo.cosmo)**self.mode_exp]
            #print(self.rho_dm_sample[0])
            self.b_sample = phys.b0*np.ones(sim.n)
            self.ne_sample = phys.ne0*np.ones(sim.n)
            self.neav = phys.ne0
            self.bav = phys.b0
            if self.dl is None:
                self.dl = cosmology.dist_luminosity(self.z,cosmo)
                self.da = cosmology.dist_angular(self.z,cosmo)
            else:
                self.da = self.dl/(1+self.z)**2
            #if(sim.num > 1):
            #    self.f_sample = np.logspace(np.log10(sim.flim[0]),np.log10(sim.flim[1]),num=sim.num)
            #else:
            #    self.f_sample = np.array([sim.nu_sb])
            self.rfarc = self.da*np.tan(radians_per_arcmin)  #radius in Mpc arcmin^-1
            self.ready = True
        else:
            self.ready = False
        return self.ready

    def setup_halo(self,sim,phys,cos_env):
        if self.check_self() and sim.check_self() and phys.check_self():
            radians_per_arcmin = 2.909e-4
            if self.dl is None:
                #set up the cosmic distances
                self.dl = cosmology.dist_luminosity(self.z,cos_env)
                self.da = cosmology.dist_angular(self.z,cos_env)
            else:
                if self.z is None:
                    self.z = 0.0
                self.da = self.dl/(1+self.z)**2
            self.rhoProfileCalc(cos_env)
            self.rfarc = self.da*np.tan(radians_per_arcmin)  #radius in Mpc arcmin^-1
            self.r_sample = [np.logspace(np.log10(self.rcore*1e-5),np.log10(2*self.rvir),sim.n),np.logspace(np.log10(self.rcore*1e-7),np.log10(2*self.rvir),sim.ngr)] #we go to 2rvir for diffusion reaons
            self.rho_dm_sample = astrophysics.rho_dm_halo(self,cos_env)#[cosmology.rho_dm(self.r_sample[0],self.rcore,self.dm,self.alpha),cosmology.rho_dm(self.r_sample[1],self.rcore,self.dm,self.alpha)]
            if(phys.lp is None):
                phys.lp = self.rcore
            if not (self.rb1 is None and self.rb2 is None):
                self.rb1Dist = self.rb1*self.rfarc/sim.theta
                self.rb2Dist = self.rb2*self.rfarc/sim.theta
            self.ne_sample = astrophysics.ne_distribution(phys,self,phys.ne_model)
            self.b_sample = astrophysics.bfield(phys,self,cos_env,phys.b_flag)
            rhoc = cosmology.rho_crit(self.z,cos_env) #critical density
            #======================================================================================
            #Substructure boosting calculations
            #======================================================================================
            if(sim.sub_mode == "sc2006"):
                rbs = 7.0
                dms = self.dm
                rhobs = substructure.rhobars(self.mvir,self.rvir,self.rcore*rbs,self.dm,self.alpha)
                rhobar = rhoc*cosmology.omega_m(self.z,cos_env)
                if dms == 4:
                    rbs = 1.0
                rho_sub = astrophysics.rho_dm(self.r_sample[0],self.rcore*rbs,dms,self.alpha)
                rho_sub_g = astrophysics.rho_dm(self.r_sample[1],self.rcore*rbs,dms,self.alpha)
                bf = substructure.delta(self.mvir,self.sub_frac,self.z,cos_env,dms,self.alpha)
                self.rho_dm_sample[0] = astrophysics.rho_boost(self.rho_dm_sample[0],rho_sub,self.rhos,rhobar,rhobs,rhoc,bf,self.sub_frac,self.mode_exp)
                self.rho_dm_sample[1] = astrophysics.rho_boost(self.rho_dm_sample[1],rho_sub_g,self.rhos,rhobar,rhobs,rhoc,bf,self.sub_frac,self.mode_exp)
            elif(sim.sub_mode == "prada"):
                rbs = 7.0
                #boost = substructure.b_sanchez(self.mvir,1.0e-6,self.z,cos_env)
                self.boost = substructure.b_p12(self.mvir,cos_env) #find boost factor from Prada 2013
                self.rho_dm_sample[0] = self.boost**(0.5*self.mode_exp)*(self.rho_dm_sample[0]*self.rhos*rhoc)**self.mode_exp #Correct rho^2 by applying boost
                self.rho_dm_sample[1] = self.boost**(0.5*self.mode_exp)*(self.rho_dm_sample[1]*self.rhos*rhoc)**self.mode_exp
                self.radio_boost = self.boost #set radio boost equal to normal
                if sim.radio_boost_flag != 0: #this flag indicates we care about the effect of sub-halo radial dist on radio boosting
                    radio_power = substructure.radio_power_ratio(self,phys,sim) #ratio of radio power at r to power at r = 0
                    rho_sub = astrophysics.rho_dm(self.r_sample[0],self.rcore*rbs,self.dm,self.alpha) #sub-halo r probability distribution
                    rho_ns = 1.0/tools.Integrate(rho_sub,self.r_sample[0]) #find normalisation factor
                    rho_sub = rho_sub*rho_ns #normalises rho_sub to integrate to 1
                    #integrate boost modified by radio power ratio at r with sub-halo position probability dist 
                    #this will give us the average synchrotron boosting factor for a sub-halo distribution
                    self.radio_boost = tools.Integrate((1.0 + (self.boost-1.0)*radio_power)*rho_sub,self.r_sample[0])
            else:
                #no substructure boosting
                self.rho_dm_sample[0] = (self.rho_dm_sample[0]*self.rhos*rhoc)**self.mode_exp
                self.rho_dm_sample[1] = (self.rho_dm_sample[1]*self.rhos*rhoc)**self.mode_exp
            #======================================================================================
            if self.weights is None or self.weights == "rho":
                weights = self.rho_dm_sample[0] #the average is weighted
            elif self.weights == "flat":
                weights = np.ones(len(self.rho_dm_sample[0]),dtype=np.float64)
            if phys.b_flag == "flat" and self.bav == 0.0:
                self.bav = phys.b0
            elif self.bav == 0.0: #check it wasn't set by input
                self.bav = tools.weightedVolAvg(self.b_sample,self.r_sample[0],weights,self.rvir)
            else:
                self.bav_flag = 1
            if self.neav == 0.0:
                self.neav = tools.weightedVolAvg(self.ne_sample,self.r_sample[0],weights,self.rvir)
            else:
                self.neav_flag = 1
            if phys.d0 is None:
                electron.diffusion_constant(self,phys)
            if self.J is None: #calculate a jfactor just for interest sake
                #junits = 4.428e-9 #from M_sun^2 Mpc^-5 to Gev^2 cm^-5
                #self.J = junits*integrate(self.rho_dm_sample[0]*self.r_sample[0]**2/(self.dl**2+self.r_sample[0]**2),self.r_sample[0])
                self.J = astrophysics.jfactor(self,cos_env)
            if self.Dfactor is None:
                self.Dfactor = astrophysics.dfactor(self,cos_env)
            self.ready = True
        else:
            self.ready = False

        return self.ready

    def haloFromHeader(self,hdr):
        #need: boost, radio_boost
        self.r_sample[0] = np.array(hdr['CRSET3'].split(),dtype=np.float64)
        self.dl = np.float64(hdr['DLUM'])
        self.da = np.float64(hdr['DANG'])
        self.profile = hdr['DMPROF']
        if self.profile == "gnfw":
            self.dm = np.float64(hdr['DMGNFW'])
        else:
            self.dm = int(self.profileDict[self.profile])
            if self.dm == -1:
                self.alpha = hdr["EINALPHA"]
        self.weights = hdr['HALOWTS']
        self.name = hdr['HNAME']
        self.mode = hdr['WMODE']
        if self.mode == "decay":
            self.mode_exp = 1.0
        else:
            self.mode_exp = 2.0
        self.mvir = np.float64(hdr['MVIR'])
        self.cvir = np.float64(hdr['CVIR'])
        self.rvir = np.float64(hdr['RVIR'])
        self.rcore = np.float64(hdr['DMSCALE'])
        self.rho0 = np.float64(hdr['DMRHO0'])
        self.J_flag = int(hdr['JNORM'])
            
#cosmo_spec = [('h',float32),('w_m',float32),('w_l',float32),('w_dm',float32),('n',float32),('w_nu',float32),('N_nu',float32),('sigma_8',float32),('w_b',float32),('G_newton',float32),('w_k',float32),('universe',np.array(char,1d,A)]     
class cosmology_env:
    """
    Container for cosmology parameters
        ---------------------------
        Attributes
        ---------------------------
        h        - reduced Hubble constant [100 km s^-1 Mpc^-1] (float)
        w_m      - matter fraction (float)
        w_l      - Lambda fraction (float)
        w_dm     - dark matter fraction (float)
        n        - power spectrum spctral index (float)
        w_nu     - neutrino fraction (float)
        N_nu     - number of neutrino species (int)
        sigma_8  - mater power spectrum normalisation (float)
        w_b      - baryon fraction (float)
        universe - 'flat' or 'curved' (String)
        w_k      - curvature fraction (float)
        cosmo    - dictionary of attributes
    """
    #default parameters are those of a flat PLANCK universe -> 1502.01589 
    def __init__(self,h=0.6774,w_m=0.3089,w_l=0.6911,uni="flat",w_b=0.0486,w_dm=0.2589,w_nu=0.0,N_nu=0,n=0.968,sigma_8=0.8159):
        self.h = h #reduce hubble constant in 100 km s^-1 Mpc^-1
        self.w_m = w_m #matter fraction
        self.w_l = w_l #lambda fraction
        self.w_dm = w_dm #dakr matter fraction
        self.n = n #matter pertubation power spectrum index
        self.w_nu = w_nu #neutrino fraction
        self.N_nu = N_nu #number of neutrino species
        self.sigma_8 = sigma_8 #matter power spectrum normalisation
        self.w_b = w_b #baryon fraction
        self.universe = uni #curved or flat
        self.setup()

    def setup(self):
        self.G_newton = 6.67408e-11*2e30*3.24078e-23**3 #Mpc^3 Msol^-1 s^-2
        if self.universe == "flat":
            self.w_k = 0 #curvature fraction
            if self.w_l + self.w_m != 1.0:
                self.w_l = 1.0 - self.w_m
        else:
            self.w_k = 1.0 - self.w_l - self.w_m
        #cosmolopy dictionary object for compatibility -> no longer in use
        self.cosmo = {'omega_M_0':self.w_m, 'omega_lambda_0':self.w_l, 'omega_k_0':self.w_k, 'h':self.h, 'omega_dm_0':self.w_dm, 'omega_b_0':self.w_b, 'sigma_8':self.sigma_8, 'n':self.n, 'omega_n_0':self.w_nu, 'N_nu':self.N_nu, 'EW':np.array([2e11,107,1e15]),'QCD':np.array([2e8,55,1e12]),'EE':np.array([0.51e6,10.8,2e9])}

    def cosmoFromHeader(self,hdr):
        self.h = hdr['COSMOH']
        self.w_dm = hdr['COSMOWDM']
        self.w_l = hdr['COSMOWL']
        self.w_m = hdr['COSMOWM']
        self.w_b = hdr['COSMOWB']
        self.w_nu = hdr['COSMOWNU']
        self.n = hdr['COSMON']
        self.N_nu = hdr['COSMONNU']
        self.sigma_8 = hdr['COSMOSIG']
        self.universe = hdr['COSMOUNI']
        self.setup()
#loop_spec = [('mmin',float32),('mmax',float32),('zmin',float32),('zmax',float32),('zn',float32),('nu',float32),('phys
#old project, don't worry about this bit
class loop_env:
    def __init__(self,fs=0.0,z=None,zmin=None,zmax=None,zn=1,nloop=9,dmmod=1,alpha=None,mmin=1e7,mmax=1e15,nu=1000,phys=None,cosm=None,sim=None):
        self.mmin = mmin
        self.mmax = mmax
        self.zmin = zmin
        self.zmax = zmax
        self.zn = zn
        self.nu = nu
        self.phys = phys
        self.cosm = cosm
        self.sim = sim
        self.mn = nloop
        self.m_sample = np.logspace(np.log10(mmin),np.log10(mmax),num=self.mn)
        self.ne_sample = None
        self.halos = None
        self.phys_set = None
        self.sub_frac = fs
        self.z = z
        self.dm = dmmod
        self.alpha = alpha

    def setup(self,z_index=0):
        if(self.phys is None or self.cosm is None or self.sim is None):
            check = False
        else:
            self.m_sample = np.logspace(np.log10(self.mmin),np.log10(self.mmax),num=self.mn)
            if(self.zmin != None and self.zmax != None and self.zn != 1):
                self.z_sample = np.logspace(np.log10(self.zmin),np.log10(self.zmax),num=self.zn)
            else:
                self.z_sample = [self.z]
                self.zn = 1
            try:
                self.z = self.z_sample[z_index]
            except IndexError:
                tools.fatal_error("Z index for loop is larger than allowed, please specify limits properly")
            mass_set_0 = np.logspace(7,15,num=9)
            ne_set_0 = np.array([1.0e-6,1.0e-6,1.0e-5,1.0e-5,1.0e-4,1.0e-4,1.0e-3,1.0e-3,1.0e-3])
            nef = sp.interp1d(mass_set_0,ne_set_0)
            ne_set = np.zeros(len(self.m_sample))
            for i in range(0,len(self.m_sample)):
                if(self.m_sample[i] < mass_set_0[0]):
                    ne_set[i] = ne_set_0[0]
                elif(self.m_sample[i] > mass_set_0[len(mass_set_0)-1]):
                    ne_set[i] = ne_set_0[len(ne_set_0)-1]
                else:
                    ne_set[i] = nef(self.m_sample[i])
            self.phys_set = []
            self.halos = []
            phys = self.phys
            for i in range(0,self.mn):
                if(self.m_sample[i] < 1e9):
                    hqb = 0.0
                    hqe = 0.0
                else:
                    hqp = phys.qb
                    hqe = phys.qe
                self.phys_set.append(physical_env(b0=phys.b0,qb=hqb,qe=hqe,ne0=ne_set[i],channel=phys.channel,mx=phys.mx,lc=phys.lc,delta=phys.delta,diff=phys.diff))
                self.phys_set[i].spectrum = phys.spectrum
                self.phys_set[i].gamma_spectrum = phys.gamma_spectrum
            for m in self.m_sample:
                self.halos.append(halo_env(m=m,z=self.z,dmmod=self.dm,alpha=self.alpha,fs=self.sub_frac)) 
            #self.sim.num = 1
            #self.sim.flim[0] = self.sim.nu_sb
            #self.sim.flim[1] = self.sim.nu_sb
            for i in range(0,self.mn):
                self.halos[i].setup(self.sim,self.phys_set[i],self.cosm)
                #print "halo: "+str(i)+" ready"
            check = True
        return check
            
        
        
