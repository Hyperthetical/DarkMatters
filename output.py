import numpy as np
from os.path import join
import sys,os

try:
    from wimp_tools import tools,cosmology_env,simulation_env,physical_env,loop_env,halo_env #wimp handling package
except:
    import wimp_tools.cosmology_env as cosmology_env
    import wimp_tools.tools as tools
    import wimp_tools.simulation_env as simulation_env
    import wimp_tools.physical_env as physical_env
    import wimp_tools.loop_env as loop_env
    import wimp_tools.halo_env as halo_env
try:
    from emm_tools import electron,radio,high_e,neutrino #emmisivity package
except:
    import emm_tools.electron as electron
    import emm_tools.radio as radio
    import emm_tools.high_e as high_e
    import emm_tools.neutrino as neutrino

class calculation:
    def __init__(self,halo,phys,sim,cosmo):
        self.halo = halo
        self.phys = phys
        self.cosmo = cosmo
        self.sim = sim
        self.yData = None
        self.xData = None
        if not self.sim.nu_flavour == "all":
            flavourStr = self.sim.nu_flavour
        else:
            flavourStr = ""
        self.allXUnits = ["hz","ev"]
        self.allYUnits = ["erg","ev","jy"]
        self.prefix = {"hz":{"k":1e3,"m":1e6,"g":1e9,"t":1e12,"p":1e15},"ev":{"k":1e3,"m":1e6,"g":1e9,"t":1e12,"p":1e15},"jy":{"m":1e-3,"u":1e-6}}
        self.outLabel = {"gflux":"gamma","hflux":"he","rflux":"radio","flux":"multi","jflux":"he_jflux","nuflux":"nu_"+flavourStr,"nu_jflux":"nu_"+flavourStr+"_jflux"}

    def calcFlux(self,calcMode,regionFlag="full",full_id=True,suppress_output=False):
        halo_id = getCalcID(self.sim,self.phys,self.cosmo,self.halo,short_id=(not full_id))
        fluxFile = halo_id+"_"+self.outLabel[calcMode]
        if regionFlag == "theta":
            fluxFile += "_"+str(self.sim.theta)+"arcminflux.out"
        elif regionFlag == "r_integrate":
            fluxFile += "_"+str(self.sim.rintegrate)+"mpcflux.out"
        elif regionFlag == "full":
            fluxFile += "_virflux.out"
        fluxData = self.__calcFlux(calcMode,regionFlag,suppress_output=suppress_output)
        erg = fluxData*self.sim.f_sample*1e-17
        write = [];write.append(self.sim.f_sample);write.append(fluxData);write.append(erg)
        self.calcWrite(log=join(self.sim.out_dir,fluxFile))
        writeFile(join(self.sim.out_dir,fluxFile),write,3,append=True)

    def __calcFlux(self,fluxMode,regionFlag,suppress_output=False):
        """
        Calculate and store multi-frequency flux at all frequencies
            ---------------------------
            Parameters
            ---------------------------
            halo       - Required : halo environment(halo_env)
            sim        - Required : simulation environment (simulation_env)
            regionFlag - Required : integration region flag 
            ---------------------------
            Output
            ---------------------------
            None - assigned to halo.radio_arcflux or halo.radio_arcflux
        """ 
        fRatio = self.__jNormSelf(fluxMode,regionFlag)
        if fluxMode == "rflux" or fluxMode == "flux":
            if self.halo.radio_emm is None:
                self.__calcEmm(fluxMode)
            print("=========================================================")
            print("Calculating Synchrotron Flux")
            print("=========================================================")
            if regionFlag == "theta":
                print("Finding Flux Within angular radius of: "+str(self.sim.theta)+" arcmin")
                radioFlux = radio.radio_flux(self.halo.da*np.tan(self.sim.theta*2.90888e-4),self.halo,self.sim)
            elif regionFlag == "r_integrate" and not self.sim.rintegrate is None:
                print("Finding Flux Within radius of: "+str(self.sim.rintegrate)+" Mpc")
                radioFlux = radio.radio_flux(self.sim.rintegrate,self.halo,self.sim)
            else:
                print("Finding Flux within Virial Radius")
                radioFlux = radio.radio_flux(self.halo.rvir,self.halo,self.sim)
            print('Magnetic Field Average Strength: '+str(self.halo.bav)+" micro Gauss")
            print('Gas Average Density: '+str(self.halo.neav)+' cm^-3')
            print("Process Complete")
            if fluxMode == "rflux":
                return radioFlux*fRatio
        if fluxMode in ["hflux","gflux","jflux","flux"]:
            if fluxMode == "jflux":
                if not suppress_output:
                    print("=========================================================")
                    print("Calculating Pion-decay Flux from J-Factor")
                    print("=========================================================")
                return high_e.gamma_from_j(self.halo,self.phys,self.sim)**4.14e-24*1.6e20
            if self.halo.he_emm is None:
                self.__calcEmm(fluxMode)
            if fluxMode == "gflux":
                gammaFlag = 0
            else:
                gammaFlag = 1
            if not suppress_output:
                print("=========================================================")
                if fluxMode == "gflux":
                    print("Calculating Pion-decay Flux")
                else:
                    print("Calculating Pion-decay, IC, and Bremsstrahlung Flux")
                print("=========================================================")
            high_e.gamma_source(self.halo,self.phys,self.sim)
            if regionFlag == "theta":
                if not suppress_output:
                    print("Finding Flux Within angular radius of: "+str(self.sim.theta)+" arcmin")
                hFlux = high_e.high_E_flux(self.halo.da*np.tan(self.sim.theta*2.90888e-4),self.halo,self.sim,gammaFlag)
            elif regionFlag == "r_integrate" and not self.sim.rintegrate is None:
                if not suppress_output:
                    print("Finding Flux Within radius of: "+str(self.sim.rintegrate)+" Mpc")
                hFlux = high_e.high_E_flux(self.sim.rintegrate,self.halo,self.sim,gammaFlag)
            else:
                if not suppress_output:
                    print("Finding Flux within Virial Radius")
                hFlux = high_e.high_E_flux(self.halo.rvir,self.halo,self.sim,gammaFlag)
            if not suppress_output and not fluxMode == "gflux":
                print('Magnetic Field Average Strength: '+str(self.halo.bav)+" micro Gauss")
                print('Gas Average Density: '+str(self.halo.neav)+' cm^-3')
                print("Process Complete")
            if not fluxMode == "flux":
                return hFlux*fRatio
        if "nu" in fluxMode: 
            if (not "jflux" in fluxMode) and self.halo.nu_emm is None:
                self.__calcEmm(fluxMode)
            print("=========================================================")
            print("Calculating Neutrino Flux")
            print("=========================================================")
            if "jflux" in fluxMode:
                nuFlux = neutrino.nu_from_j(self.halo,self.phys,self.sim)*4.14e-24*1.6e20
            else:
                if regionFlag == "theta":
                    print("Finding Flux Within angular radius of: "+str(self.sim.theta)+" arcmin")
                    nuFlux = neutrino.nu_flux(self.halo.da*np.tan(self.sim.theta*2.90888e-4),self.halo,self.sim,0)
                elif regionFlag == "r_integrate" and not self.sim.rintegrate is None:
                    print("Finding Flux Within radius of: "+str(self.sim.rintegrate)+" Mpc")
                    nuFlux = neutrino.nu_flux(self.sim.rintegrate,self.halo,self.sim,0)
                else:
                    print("Finding Flux within Virial Radius")
                    nuFlux = neutrino.nu_flux(self.halo.rvir,self.halo,self.sim,0)
            print("Process Complete")
            return nuFlux*fRatio
        return hFlux*fRatio + radioFlux*fRatio

    def __calcEmm(self,fluxMode):
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
        if fluxMode in ["rflux","flux"]:
            if self.halo.electrons is None:
                self.__calcElectrons()
            if not self.sim.radio_emm_from_c:
                print("=========================================================")
                print("Calculating Radio Emissivity Functions with Python")
                print("=========================================================")
                self.halo.radio_emm = radio.radio_emm(self.halo,self.phys,self.sim)
            else:
                print("=========================================================")
                print("Calculating Radio Emissivity Functions with C")
                print("=========================================================")
                py_file = "temp_radio_emm_py.out"
                c_file = "temp_radio_emm_c.in"
                wd = os.getcwd()
                self.halo.radio_emm = radio.emm_from_c(join(wd,py_file),join(wd,c_file),self.halo,self.phys,self.sim)
                os.remove(join(wd,py_file))
                os.remove(join(wd,c_file))
                if self.halo.radio_emm is None:
                    tools.fatal_error("The radio_emm executable is not compiled/has no specified location")
            print("Process Complete")
        if fluxMode in ["hflux","flux"]:
            if self.halo.electrons is None:
                self.__calcElectrons()
            print("=========================================================")
            print("Calculating High-Energy Emissivity Functions")
            print("=========================================================")
            self.halo.he_emm = high_e.high_E_emm(self.halo,self.phys,self.sim)
            print("Process Complete")
        if fluxMode == "nuflux":
            print("=========================================================")
            print("Calculating Neutrino Emissivity Functions")
            print("=========================================================")
            self.halo.nu_emm = neutrino.nu_emm(self.halo,self.phys,self.sim)
            print("Process Complete")

    
    def __calcElectrons(self):
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
        if not self.sim.electrons_from_c:
            print("=========================================================")
            print("Calculating Electron Equilibriumn Distributions with Python")
            print("=========================================================")
            print('Magnetic Field Average Strength: '+str(self.halo.bav)+" micro Gauss")
            print('Gas Average Density: '+str(self.halo.neav)+' cm^-3')
            self.halo.electrons = electron.equilibrium_p(self.halo,self.phys)
        else:
            print("=========================================================")
            print("Calculating Electron Equilibriumn Distributions with C")
            print("=========================================================")
            print('Magnetic Field Average Strength: '+str(self.halo.bav)+" micro Gauss")
            print('Gas Average Density: '+str(self.halo.neav)+' cm^-3')
            py_file = "temp_electrons_py.out"
            c_file = "temp_electrons_c.in"
            wd = os.getcwd()
            self.halo.electrons = electron.electrons_from_c(join(wd,py_file),join(wd,c_file),self.halo,self.phys,self.sim)
            os.remove(join(wd,py_file))
            os.remove(join(wd,c_file))
            if self.halo.electrons is None:
                tools.fatal_error("The electron executable is not compiled/location not specified")
        print("Process Complete")

    def __jNormSelf(self,fluxMode,regionFlag):
        if self.halo.J_flag != 0 and (not "jflux" in fluxMode):
            hfmax = self.phys.mx/(1e6*4.136e-15*1e-9) #MHz
            hsim = simulation_env(n=self.sim.n,ngr=self.sim.ngr,num=20,fmin=1e-3*hfmax,fmax=0.1*hfmax,theta=self.sim.theta,nu_sb=self.sim.nu_sb)
            jhalo = halo_env()
            attSet = [att for att in dir(self.halo) if (not att.startswith("__")) and (not att in ["physical_averages","setup_halo","setup","setup_ucmh","check_self","make_spectrum"])]
            for att in attSet:
                if not att in ["mode","mode_exp","rho_dm_sample"]:
                    setattr(jhalo,att,getattr(self.halo,att))
            jcheck = jhalo.setup(self.sim,self.phys,self.cos_env)
            hsim.sample_f()
            gflux = self.__calcFlux("gflux",regionFlag,suppress_output=True)
            jflux = self.__calcFlux("jflux",regionFlag,suppress_output=True)
            fRatio = (sum(jflux/gflux)/len(jflux))**(0.5*self.halo.mode_exp)
            self.sim.jnormed = True
            print("Normalisation factor: "+str(fRatio))
            return fRatio
        else:
            return 1.0

    def calcWrite(self,writeMode="flux",log=None):
        """
        Write calculation data to a target output
            ---------------------------
            Parameters
            ---------------------------
            log        - Required : log file name (if None uses stdout) (String or None)
            writeMode - Required : 'flux' displays all information (String)
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
        outstream.write(prefix+"Output directory: "+self.sim.out_dir+end)
        if self.halo.name != None:
            outstream.write(prefix+'Halo Name: '+str(self.halo.name)+end)
        elif(writeMode == "flux" and self.halo.ucmh == 0 and (not self.halo.mvir is None)):
            outstream.write(prefix+'Halo Mass Code: m'+str(int(np.log10(self.halo.mvir)))+end)
        #outstream.write(prefix+'Field File Code: b'+str(int(phys.b0))+"q"+str(phys.qb)+end)
        outstream.write(prefix+"Frequency Samples: "+str(self.sim.num)+end)
        outstream.write(prefix+"Minimum Frequency Sampled: "+str(self.sim.flim[0])+" MHz"+end)
        outstream.write(prefix+"Maximum Frequency Sampled: "+str(self.sim.flim[1])+" MHz"+end)
        outstream.write(prefix+"Radial Grid Intervals: "+str(self.sim.n)+end)
        outstream.write(prefix+"Green's Function Grid Intervals: "+str(self.sim.ngr)+end)
        outstream.write(prefix+'Minimum Sampled Radius: '+str(self.halo.r_sample[0][0])+' Mpc'+end)
        outstream.write(prefix+'Maximum Sampled Radius: '+str(self.halo.r_sample[0][self.sim.n-1])+' Mpc'+end)
        outstream.write(prefix+'======================================================'+end)
        outstream.write(prefix+'Dark Matter Parameters: '+end)
        outstream.write(prefix+'======================================================'+end)
        outstream.write(prefix+'WIMP mass: '+str(self.phys.mx)+' GeV'+end)
        if self.halo.mode_exp == 2.0 and not self.halo.mode == "special":
            outstream.write(prefix+'Annihilation channels used: '+str(self.phys.channel)+end)
        else:
            outstream.write(prefix+'Decay channels used: '+str(self.phys.channel)+end)
        if not self.halo.mode == "special":
            outstream.write(prefix+'Branching ratios used: '+str(self.phys.branching)+end)
        outstream.write(prefix+"Particle physics model label: "+str(self.phys.particle_model)+end)
        outstream.write(prefix+'======================================================'+end)
        outstream.write(prefix+'Halo Parameters: '+end)
        outstream.write(prefix+'======================================================'+end)
        outstream.write(prefix+'Redshift z: '+str(self.halo.z)+end)
        if self.halo.ucmh == 0:
            if(self.halo.dm == 1):
                outstream.write(prefix+'Halo Model: NFW'+end)
            elif(self.halo.dm == 2):
                outstream.write(prefix+'Halo Model: Cored (Burkert)'+end)
            elif(self.halo.dm == -1):
                outstream.write(prefix+"Halo Model: Einasto, Alpha: "+str(self.halo.alpha)+end)
            elif(self.halo.dm == 3):
                outstream.write(prefix+'Halo Model: Cored (Isothermal)'+end)
        else:
            outstream.write(prefix+'Halo Model: Ultra-compact'+end)
            #outstream.write(prefix+'Phase Transition: '+halo.phase+end)
        if(writeMode == "flux"):
            outstream.write(prefix+'Virial Mass: '+str(self.halo.mvir)+" Solar Masses"+end)
            if not self.halo.Dfactor is None:
                outstream.write(prefix+'Dfactor: '+str(self.halo.Dfactor)+" GeV cm^-2"+end)
            if not self.halo.J is None:
                outstream.write(prefix+'Jfactor: '+str(self.halo.J)+" GeV^2 cm^-5"+end)
                if self.sim.jnormed:
                    outstream.write(prefix+'Normalised rho to Jfactor'+end)
            if self.halo.ucmh == 0:
                outstream.write(prefix+'Rho_s/Rho_crit: '+str(self.halo.rhos)+end)
            outstream.write(prefix+'Virial Radius: '+str(self.halo.rvir)+' Mpc'+end)
            if self.halo.ucmh == 0:
                outstream.write(prefix+'Virial Concentration: '+str(self.halo.cvir)+end)
                outstream.write(prefix+'Core Radius: '+str(self.halo.rcore)+' Mpc'+end)
            outstream.write(prefix+'Luminosity Distance: '+str(self.halo.dl)+' Mpc'+end)
        outstream.write(prefix+'Angular Diameter Distance: '+str(self.halo.da)+' Mpc'+end)
        outstream.write(prefix+'Angular Observation Radius Per Arcmin: '+str(self.halo.rfarc)+' Mpc arcmin^-1'+end)
        if self.sim.theta > 0.0 and not self.sim.theta is None:  
            outstream.write(prefix+'Observation Radius for '+str(self.sim.theta)+' arcmin is '+str(self.halo.da*np.tan(self.sim.theta*2.90888e-4))+" Mpc"+end)
        if not self.sim.rintegrate is None:  
            outstream.write(prefix+'Observation Radius r_integrate '+str(self.sim.rintegrate)+" Mpc"+end)
        outstream.write(prefix+'======================================================'+end)
        outstream.write(prefix+'Gas Parameters: '+end)
        outstream.write(prefix+'======================================================'+end)
        if self.phys.ne_model == "flat":
            outstream.write(prefix+'Gas Distribution: '+"Flat Profile"+end)
            outstream.write(prefix+'Gas Central Density: '+str(self.phys.ne0)+' cm^-3'+end)
        elif(self.phys.ne_model == "powerlaw" or self.phys.ne_model == "pl"):
            outstream.write(prefix+'Gas Distribution: '+"Power-law profile"+end)
            outstream.write(prefix+'Gas Central Density: '+str(self.phys.ne0)+' cm^-3'+end)
            if self.phys.lb is None:
                outstream.write(prefix+'Scale radius: '+str(self.halo.rcore*1e3)+" kpc"+end)
            else:
                outstream.write(prefix+'Scale radius: '+str(self.phys.lb*1e3)+" kpc"+end)
            outstream.write(prefix+'PL Index: '+str(-1*self.phys.qe)+end)
        elif(self.phys.ne_model == "king"):
            outstream.write(prefix+'Gas Distribution: '+"King-type profile"+end)
            outstream.write(prefix+'Gas Central Density: '+str(self.phys.ne0)+' cm^-3'+end)
            if self.phys.lb is None:
                outstream.write(prefix+'Scale radius: '+str(self.halo.rcore*1e3)+" kpc"+end)
            else:
                outstream.write(prefix+'Scale radius: '+str(self.phys.lb*1e3)+" kpc"+end)
            outstream.write(prefix+'PL Index: '+str(-1*self.phys.qe)+end)
        elif(self.phys.ne_model == "exp"):
            outstream.write(prefix+'Gas Distribution: '+"Exponential"+end)
            outstream.write(prefix+'Gas Central Density: '+str(self.phys.ne0)+' cm^-3'+end)
            outstream.write(prefix+'Scale radius: '+str(self.halo.r_stellar_half_light*1e3)+" kpc"+end)

        if(writeMode == "flux"):
            outstream.write(prefix+'Gas Average Density (rvir): '+str(self.halo.neav)+' cm^-3'+end)
        outstream.write(prefix+'======================================================'+end)
        outstream.write(prefix+'Halo Substructure Parameters: '+end)
        outstream.write(prefix+'======================================================'+end)
        if(self.sim.sub_mode == "sc2006"):
            outstream.write(prefix+"Substructure Mode: Colafrancesco 2006"+end)
            outstream.write(prefix+"Substructure Fraction: "+str(self.halo.sub_frac)+end)
        elif(self.sim.sub_mode == "none" or self.sim.sub_mode is None):
            outstream.write(prefix+"Substructure Mode: No Substructure"+end)
        elif(self.sim.sub_mode == "prada"):
            outstream.write(prefix+"Substructure Mode: Sanchez-Conde & Prada 2013"+end)
            outstream.write(prefix+"Boost Factor: "+str(self.halo.boost)+end)
            outstream.write(prefix+"Synchrotron Boost Factor: "+str(self.halo.radio_boost)+end)
        outstream.write(prefix+'======================================================'+end)
        outstream.write(prefix+'Magnetic Field Parameters: '+end)
        outstream.write(prefix+'======================================================'+end)
        if(self.phys.b_flag == "flat"):
            outstream.write(prefix+'Magnetic Field Model: '+"Flat Profile"+end)
        elif(self.phys.b_flag == "powerlaw" or self.phys.b_flag == "pl"):
            outstream.write(prefix+'Magnetic Field Model: '+"Power-law profile"+end)
            outstream.write(prefix+'PL Index: '+str(-1*self.phys.qb*self.phys.qe)+end)
        elif(self.phys.b_flag == "follow_ne"):
            outstream.write(prefix+'Magnetic Field Model: '+"Following Gas Profile"+end)
            outstream.write(prefix+'PL Index on n_e: '+str(self.phys.qb)+end)
        elif(self.phys.b_flag == "equipartition"):
            outstream.write(prefix+'Magnetic Field Model: '+"Energy Equipartition with Gas"+end)
        elif(self.phys.b_flag == "sc2006"):
            outstream.write(prefix+'Magnetic Field Model: '+"Two-Parameter Coma Profile"+end)
            outstream.write(prefix+'Magnetic Field Scaling Radii: '+str(self.halo.rb1)+" Mpc "+str(self.halo.rb2)+" Mpc"+end)
        elif(self.phys.b_flag == "exp"):
            outstream.write(prefix+'Magnetic Field Model: '+"Exponential"+end)
            if self.phys.qb == 0.0:
                outstream.write(prefix+'Scale radius: '+str(self.halo.r_stellar_half_light*1e3)+" kpc"+end)
            else:
                outstream.write(prefix+'Scale radius: '+str(self.phys.qb*1e3)+" kpc"+end)
        elif(self.phys.b_flag == "m31"):
            outstream.write(prefix+'Magnetic Field Model: '+"M31"+end)
            outstream.write(prefix+'Scale radius r1: '+str(self.phys.qb*1e3)+" kpc"+end)
        elif(self.phys.b_flag == "m31exp"):
            outstream.write(prefix+'Magnetic Field Model: '+"M31 + Exponential after 14 kpc"+end)
            outstream.write(prefix+'Scale radius r1: '+str(self.phys.qb*1e3)+" kpc"+end)
        outstream.write(prefix+'Magnetic Field Strength Parameter: '+str(self.phys.b0)+' micro Gauss'+end)
        outstream.write(prefix+'Magnetic Field Average Strength (rvir): '+str(self.halo.bav)+" micro Gauss"+end)
        if(self.phys.diff == 0):
            outstream.write(prefix+'No Diffusion'+end)
        else:
            outstream.write(prefix+'Spatial Diffusion'+end)
            outstream.write(prefix+'Turbulence scale: '+str(self.phys.lc)+' kpc'+end)
            outstream.write(prefix+'Turbulence Index: '+str(self.phys.delta)+end)
            outstream.write(prefix+'Diffusion constant: '+str(self.phys.d0)+" cm^2 s^-1"+end)
        if not log is None:
            outstream.close()

def writeFile(file_name,data,cols,index_row=0,append=False):
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
        tools.fatal_error("I/O Error: Could not open "+file_name+" for writing")
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

def getCalcID(sim,phys,cos_env,halo,noBfield=False,noGas=False,short_id=False):
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

    if not halo.name is None:
        halo_str = halo.name
    else:
        halo_str = "m"+str(int(np.log10(halo.mvir)))
    halo_str += "_"

    wimp_str = phys.particle_model+"_mx"+str(phys.mx)+"GeV"
    if halo.mode == "decay":
        wimp_str += "_decay"
    if short_id:
        return wimp_str
    wimp_str += "_"

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


    if halo.ucmh == 0:
        file_id = halo_str+wimp_str+z_str+field_str+diff_str+dm_str+sub_str[:-1]
    else:
        file_id = halo_str+wimp_str+z_str+field_str+diff_str[:-1]
    return file_id


