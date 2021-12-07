import numpy as np
from astropy.io import fits 
from wimp_tools import tools,cosmology
import pickle
    

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

def getCalcID(sim,phys,cos_env,halo,noBfield=False,noGas=False,short_id=False,noMass=False):
    """
    Builds an output file id code
        ---------------------------
        Parameters
        ---------------------------
        sim       - Required : simulation environment (simulation_env)
        phys      - Required : physical environment (phys_env)
        cos_env   - Required : cosmology environment (cosmology_env)
        halo      - Required : halo environment(halo_env)
        noBfield  - Optional : if True leave out the B field model details
        noGas     - Optional : if True leave out the gas model details
        short_id  - Optional : if True include only halo label and WIMP model details
        ---------------------------
        Output
        ---------------------------
        Unique file ID prefix (string)
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
    if halo.weights == "flat":
        sub_str += "weights-flat_"
    else:
        sub_str += "weights-rhosq_"

    z_str = "z"+str(halo.z)+"_"  #redshift value to append to file names

    if not noBfield:
        if(phys.diff == 0):
            diff_str = "d0_"
        else:
            diff_str = "d1_{:02.1e}_".format(phys.d0)
    else:
        diff_str = ""

    if not halo.name is None:
        halo_str = halo.name
        z_str = ""
    else:
        halo_str = "m"+str(int(np.log10(halo.mvir)))
    halo_str += "_"
    try:
        mxSplit = str(phys.mx).split(".")
        if float(mxSplit[-1]) == 0.0:
            mxStr = str(int(phys.mx))
        else:
            mxStr = str(phys.mx)
    except:
        mxStr = str(phys.mx) 

    if not noMass:
        wimp_str = phys.particle_model+"_mx"+mxStr+"GeV"
    else:
        wimp_str = phys.particle_model
    if halo.mode == "decay":
        wimp_str += "_decay"
    if short_id:
        return halo_str+wimp_str
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

def fitsEmm(calcSet,fluxMode):
    emmCube = []
    if fluxMode == "rflux":
        fluxStr = "sync"
    elif fluxMode == "hflux":
        fluxStr = "gamma"
    elif fluxMode == "nuflux":
        fluxStr = "neutrino"
    for c in calcSet:
        if fluxMode == "rflux":
            emmCube.append(c.halo.radio_emm)
        elif fluxMode == "hflux":
            emmCube.append(c.halo.he_emm)
        elif fluxMode == "nuflux":
            emmCube.append(c.halo.nu_emm)
    emmCube = np.array(emmCube,dtype=np.float64)
    hdu = fits.PrimaryHDU(emmCube)
    hdr = hdu.header
    hdr = fitsHeaderCommon(calcSet[0],hdr)
    if not fluxMode in ["nuflux"]:
        hdr = fitsHeaderGB(calcSet[0],hdr)
    hdr['CRSET1'] = " ".join(str(mx) for mx in calcSet[0].sim.mx_set)
    hdr['CTYPE1'] = "WIMP Mass"
    hdr["CUNIT1"] = "GeV"
    hdr['CRVAl2'] = calcSet[0].sim.f_sample[0]
    hdr['CRPIX2'] = 0
    hdr['CDELT2'] = np.log10(calcSet[0].sim.f_sample[1]) - np.log10(calcSet[0].sim.f_sample[0]) 
    hdr['CTYPE2'] = "Frequency"
    hdr["CUNIT2"] = "MHz"
    hdr['CRSET2'] = " ".join(str(x) for x in calcSet[0].sim.f_sample)
    hdr['CRVAl3'] = calcSet[0].halo.r_sample[0][0]
    hdr['CRPIX3'] = 0
    hdr['CDELT3'] = np.log10(calcSet[0].halo.r_sample[0][1]) - np.log10(calcSet[0].halo.r_sample[0][0]) 
    hdr['CTYPE3'] = "Radius"
    hdr["CUNIT3"] = "Mpc"
    hdr['CRSET3'] = " ".join(str(x) for x in calcSet[0].halo.r_sample[0])
    hdr['DMCALC'] = fluxMode+"_emm"
    hdul = fits.HDUList([hdu])
    hdul.writeto(getCalcID(calcSet[0].sim,calcSet[0].phys,calcSet[0].cosmo,calcSet[0].halo,noMass=True)+"_"+fluxStr+"_emm.fits",overwrite=True)

def fitsSB(calcSet,fluxMode,fName=None):
    if fluxMode == "rflux":
        fluxStr = "sync"
    elif fluxMode == "hflux":
        fluxStr = "gamma"
    elif fluxMode == "nuflux":
        fluxStr = "neutrino"
    else:
        fluxStr = "flux"
    hdul = []
    fullGrid = []
    for c in calcSet:
        cGrid = []
        for nu in c.sim.f_sample:
            cGrid.append(c.calcSB(nu,fluxMode,full_id=True,suppress_output=False,writeFileFlag=False)[1])
        cGrid = np.array(cGrid,dtype=np.float64)
        fullGrid.append(cGrid)
        #print(c.calcLabel,cGrid.shape)
        if hdul == []:
            hdu = fits.PrimaryHDU(cGrid)
        else:
            hdu = fits.ImageHDU(cGrid)
        hdr = hdu.header
        hdr = fitsHeaderCommon(c,hdr)
        if not fluxMode in ["nuflux"]:
            hdr = fitsHeaderGB(c,hdr)
        hdr['CRSET1'] = " ".join(str(mx) for mx in c.sim.mx_set)
        hdr['CTYPE1'] = "WIMP Mass"
        hdr["CUNIT1"] = "GeV"
        hdr['CRVAl2'] = c.sim.f_sample[0]
        hdr['CRPIX2'] = 0
        try:
            fLogSpacing = np.log10(c.sim.f_sample[1:]) - np.log10(c.sim.f_sample[:-1])
            fLinSpacing = c.sim.f_sample[1:] - c.sim.f_sample[:-1]
            logTest = np.sum(fLogSpacing-fLogSpacing[0]) < fLogSpacing[0]*1e-4*c.sim.num
            linTest = np.sum(fLinSpacing-fLinSpacing[0]) < fLinSpacing[0]*1e-4*c.sim.num
            if logTest:
                hdr['CDELT2'] = fLogSpacing[0]
            elif linTest:
                hdr['CDELT2'] = fLinSpacing[0]
            else:
                hdr["CDELT2"] = -1
        except:
            hdr['CDELT2'] = 0.0
        hdr['CTYPE2'] = "Frequency"
        hdr["CUNIT2"] = "MHz"
        hdr['CRSET2'] = " ".join(str(x) for x in calcSet[0].sim.f_sample)
        hdr['CRVAl3'] = calcSet[0].halo.r_sample[0][0]
        hdr['CRPIX3'] = 0
        angSample = np.arctan(c.halo.r_sample[0]/np.sqrt(c.halo.dl**2+c.halo.r_sample[0]**2))*3437.75 #arcminutes
        hdr['CDELT3'] = np.log10(angSample[1]) - np.log10(angSample[0]) 
        hdr['CTYPE3'] = "Angular radius"
        hdr["CUNIT3"] = "Arcminute"
        hdr['CRSET3'] = " ".join(str(x) for x in angSample)
        hdr['DMCALC'] = fluxMode+"_emm"
        hdul.append(hdu)
    hdul = fits.HDUList(hdul)
    fullGrid = np.array(fullGrid)
    if fName is None:
        hdul.writeto(getCalcID(calcSet[0].sim,calcSet[0].phys,calcSet[0].cosmo,calcSet[0].halo,noMass=True)+"_"+fluxStr+"_sb.fits",overwrite=True)
        np.save(getCalcID(calcSet[0].sim,calcSet[0].phys,calcSet[0].cosmo,calcSet[0].halo,noMass=True)+"_"+fluxStr+"_sb.npy",fullGrid)
    else:
        hdul.writeto(fName,overwrite=True)
        np.save(getCalcID(calcSet[0].sim,calcSet[0].phys,calcSet[0].cosmo,calcSet[0].halo,noMass=True)+"_"+fluxStr+"_sb.npy",fullGrid)

def pickleSB(calcSet,fluxMode,fName=None):
    if fluxMode == "rflux":
        fluxStr = "sync"
    elif fluxMode == "hflux":
        fluxStr = "gamma"
    elif fluxMode == "nuflux":
        fluxStr = "neutrino"
    else:
        fluxStr = "flux"
    hdul = []
    fullGrid = []
    calcDict = {}
    calcDict['calcData'] = {}
    calcDict['calcData']['wimpMass'] = calcSet[0].sim.mx_set
    calcDict['calcData']['frequencies'] = calcSet[0].sim.f_sample
    calcDict['calcData']['angularRadius'] = calcSet[0].halo.r_sample[0]
    for c in calcSet:
        cGrid = []
        for nu in c.sim.f_sample:
            cGrid.append(c.calcSB(nu,fluxMode,full_id=True,suppress_output=False,writeFileFlag=False)[1])
        cGrid = np.array(cGrid,dtype=np.float64)
        fullGrid.append(cGrid)
    fullGrid = np.array(fullGrid)
    calcDict['resultData'] = fullGrid
    calcDict = pickleDictCommon(calcSet[0],calcDict)
    calcDict = pickleDictGB(calcSet[0],calcDict)
    if fName is None:
        pickle.dump(getCalcID(calcSet[0].sim,calcSet[0].phys,calcSet[0].cosmo,calcSet[0].halo,noMass=True)+"_"+fluxStr+"_sb.pkl",calcDict)
    else:
        pickle.dump(fName,calcDict)


def fitsFlux(calcSet,fluxMode,regionMode,header=None):
    jyCube = []
    if fluxMode == "rflux":
        fluxStr = "sync"
    elif fluxMode == "hflux":
        fluxStr = "gamma"
    elif fluxMode == "nuflux":
        fluxStr = "neutrino"
    else:
        fluxStr = 'flux'
    for c in calcSet:
        jyCube.append(c.calcFlux(fluxMode,regionFlag=regionMode,full_id=True,suppress_output=False))
    jyCube = np.array(jyCube,dtype=np.float64)
    hdu = fits.PrimaryHDU(jyCube)
    hdr = hdu.header
    hdr = fitsHeaderCommon(calcSet[0],hdr)
    if not fluxMode in ["nuflux"]:
        hdr = fitsHeaderGB(calcSet[0],hdr)
    if header is None:
        hdr['CRSET1'] = " ".join(str(mx) for mx in calcSet[0].sim.mx_set)
        hdr['CTYPE1'] = "WIMP Mass"
        hdr["CUNIT1"] = "GeV"
        hdr['CRVAl2'] = calcSet[0].sim.f_sample[0]
        hdr['CRPIX2'] = 0
        hdr['CDELT2'] = np.log10(calcSet[0].sim.f_sample[1]) - np.log10(calcSet[0].sim.f_sample[0]) 
        hdr['CTYPE2'] = "Frequency"
        hdr["CUNIT2"] = "MHz"
        hdr['CRSET2'] = " ".join(str(x) for x in calcSet[0].sim.f_sample)
        hdr['CRVAl3'] = calcSet[0].halo.r_sample[0][0]
        hdr['CRPIX3'] = 0
        hdr['CDELT3'] = np.log10(calcSet[0].halo.r_sample[0][1]) - np.log10(calcSet[0].halo.r_sample[0][0]) 
        hdr['CTYPE3'] = "Radius"
        hdr["CUNIT3"] = "Mpc"
        hdr['CRSET3'] = " ".join(str(x) for x in calcSet[0].halo.r_sample[0])
    else:
        hdr = header
        hdr['DMCALC'] = fluxMode
    hdul = fits.HDUList([hdu])
    hdul.writeto(getCalcID(calcSet[0].sim,calcSet[0].phys,calcSet[0].cosmo,calcSet[0].halo,noMass=True)+"_"+fluxStr+".fits",overwrite=True)

def fitsElectron(calcSet):
    eCube = []
    hdus = []
    for c in calcSet:
        eCube = c.halo.electrons
        eCube = np.array(eCube,dtype=np.float64)
        if hdus == []:
            hdu = fits.PrimaryHDU(eCube)
        else:
            hdu = fits.ImageHDU(eCube)
        hdr = hdu.header
        hdr = fitsHeaderCommon(c,hdr)
        hdr = fitsHeaderGB(c,hdr)
        hdr['CRVAl1'] = c.phys.spectrum[0][0]
        hdr['CRSET1'] = " ".join(str(ps) for ps in c.phys.spectrum[0])
        hdr['CRPIX1'] = 0
        hdr['CDELT1'] = np.log10(c.phys.spectrum[0][1]*c.phys.me) - np.log10(c.phys.spectrum[0][0]*c.phys.me) 
        hdr['CTYPE1'] = "Electron energy (log10-spaced)"
        hdr["CUNIT1"] = "GeV"
        hdr['CRVAl2'] = c.halo.r_sample[0][0]
        hdr['CRSET2'] = " ".join(str(x) for x in c.halo.r_sample[0])
        hdr['CRPIX2'] = 0
        hdr['CDELT2'] = np.log10(c.halo.r_sample[0][1]) - np.log10(c.halo.r_sample[0][0]) 
        hdr['CTYPE2'] = "Radius (log10-spaced)"
        hdr["CUNIT2"] = "Mpc"
        hdr["WIMP"] = c.phys.mx
        hdus.append(hdu)
    hdul = fits.HDUList(hdus)
    hdul.writeto(getCalcID(calcSet[0].sim,calcSet[0].phys,calcSet[0].cosmo,calcSet[0].halo,noMass=True)+"_electrons.fits",overwrite=True)

def fitsHeaderCommon(calc,hdr):
    hdr['DMPROF'] = calc.halo.profile
    if calc.halo.profile == "gnfw":
        hdr['DMGNFW'] = calc.halo.dm
    elif calc.halo.profile in ["einasto","ein"]:
        hdr['EINALPHA'] = calc.halo.alpha
    hdr['DMSCALE'] = calc.halo.rcore
    hdr['DMRHO0'] = calc.halo.rhos*cosmology.rho_crit(calc.halo.z,calc.cosmo)
    hdr['RVIR'] = calc.halo.rvir
    hdr['MVIR'] = calc.halo.mvir
    hdr['CVIR'] = calc.halo.cvir
    hdr['HALOWTS'] = calc.halo.weights
    hdr['DISTUNIT'] = "Mpc"
    hdr["MASSUNIT"] = "Msol"
    hdr['DLUM'] = calc.halo.dl
    hdr['DANG'] = calc.halo.da
    hdr['PMODEL'] = calc.phys.particle_model
    hdr['HNAME'] = calc.halo.name
    hdr['PCHAN'] = " ".join(str(ps) for ps in calc.phys.channel)
    hdr['BRANCH'] = " ".join(str(ps) for ps in calc.phys.branching)
    hdr['WMODE'] = calc.halo.mode
    hdr['JNORM'] = calc.sim.jnormed
    if(calc.sim.sub_mode == "sc2006"):
        hdr['SUBMODE'] = calc.sim.sub_mode
        hdr['SUBFRAC'] = calc.halo.sub_frac
    elif calc.sim.sub_mode == "none" or calc.sim.sub_mode is None:
        hdr['SUBMODE'] = "none"
    elif(calc.sim.sub_mode == "prada"):
        hdr['SUBMODE'] = "Sanchez-Conde & Prada 2013"
        hdr['SUBBOOST'] = calc.halo.boost
    hdr['COSMOWM'] = calc.cosmo.w_m
    hdr['COSMOWL'] = calc.cosmo.w_l
    hdr['COSMOWDM'] = calc.cosmo.w_dm
    hdr['COSMOWB'] = calc.cosmo.w_b
    hdr['COSMOUNI'] = calc.cosmo.universe
    hdr['COSMON'] = calc.cosmo.n
    hdr['COSMONNU'] = calc.cosmo.N_nu
    hdr['COSMOWNU'] = calc.cosmo.w_nu
    hdr['COSMOH'] = calc.cosmo.h
    hdr['COSMOSIG'] = calc.cosmo.sigma_8 
    return hdr

def pickleDictCommon(calc,dict):
    dict['haloData'] = {'haloName':calc.halo.name,'haloProfile':calc.halo.profile,'haloScale':calc.halo.rcore,'haloNorm':calc.halo.rhos*cosmology.rho_crit(calc.halo.z,calc.cosmo),'haloRvir':calc.halo.rvir,'haloMvir':calc.halo.mvir,'haloCvir':calc.halo.cvir,'haloWeights':calc.halo.weights}
    if calc.halo.profile == "gnfw":
        dict['haloData']['haloIndex'] = calc.halo.dm
    elif calc.halo.profile in ["einasto","ein"]:
        dict['haloData']['haloAlpha'] = calc.halo.alpha
    dict['units'] = {"mass":"Msun","dist":"mpc","freq":"MHz","energy":"GeV","angle":"arcminute"}
    dict['haloData']['haloLumDist'] = calc.halo.dl
    dict['haloData']['haloAngDist'] = calc.halo.da
    dict['particleData'] = {}
    dict['particleData']['particleModel'] = calc.phys.particle_model
    if calc.phys.model_independent == 1:
        dict['particleData']['particleChannel'] = calc.phys.channel
        dict['particleData']['particleBranching'] = calc.phys.branching
    dict['particleData']['emmMode'] = calc.halo.mode
    dict['haloData']['haloJNormed'] = calc.sim.jnormed
    if(calc.sim.sub_mode == "sc2006"):
        dict['haloData']['haloSubstructure'] = calc.sim.sub_mode
        dict['haloData']['haloSubFrac'] = calc.halo.sub_frac
    elif calc.sim.sub_mode == "none" or calc.sim.sub_mode is None:
        dict['haloData']['haloSubstructure'] = "none"
    elif(calc.sim.sub_mode == "prada"):
        dict['haloData']['haloSubstructure'] = "Sanchez-Conde & Prada 2013"
        dict['haloData']['haloSubBoost'] = calc.halo.boost
    dict['cosmoData'] = {}
    dict['cosmoData']['omega_m'] = calc.cosmo.w_m
    dict['cosmoData']['omega_lambda'] = calc.cosmo.w_l
    dict['cosmoData']['omega_dm'] = calc.cosmo.w_dm
    dict['cosmoData']['omega_b'] = calc.cosmo.w_b
    dict['cosmoData']['curvature'] = calc.cosmo.universe
    dict['cosmoData']['psIndex'] = calc.cosmo.n
    dict['cosmoData']['numberNeutrinos'] = calc.cosmo.N_nu
    dict['cosmoData']['omega_neutrino'] = calc.cosmo.w_nu
    dict['cosmoData']['h'] = calc.cosmo.h
    dict['cosmoData']['sigma8'] = calc.cosmo.sigma_8 
    return dict


def pickleDictGB(calc,hdr):
    hdr['gasData'] = {}
    hdr['magneticData'] = {}
    hdr['diffData'] = {}
    hdr['gasData']['gasProfile'] = calc.phys.ne_model
    hdr['gasData']['gasNorm'] = calc.phys.ne0
    hdr['gasData']['gasAverage'] = calc.halo.neav
    if(calc.phys.ne_model == "powerlaw" or calc.phys.ne_model == "pl" or calc.phys.ne_model == "king"):
        if calc.phys.lb is None:
            hdr['gasData']['gasScale'] = calc.halo.rcore
        else:
            hdr['gasData']['gasScale'] = calc.phys.lb
        hdr['gasData']['gasIndex'] = -1*calc.phys.qe
    elif(calc.phys.ne_model == "exp"):
        if calc.halo.r_stellar_half_light is None:
            hdr['gasData']['gasScale'] = calc.halo.phys.lb
        else:
            hdr['gasData']['gasScale'] = calc.halo.r_stellar_half_light
    hdr['magneticData']['magneticProfile'] = calc.phys.b_flag
    hdr['magneticData']['magneticNorm'] = calc.phys.b0
    hdr['magneticData']['magneticAverage'] = calc.halo.bav
    if(calc.phys.b_flag == "powerlaw" or calc.phys.b_flag == "pl"):
        hdr['magneticData']['magneticIndex'] = -1*calc.phys.qb
    elif(calc.phys.b_flag == "follow_ne"):
        hdr['magneticData']['magneticIndex'] = calc.phys.qb
    elif calc.phys.b_flag == "sc2006":
        hdr['magneticData']['magneticScale1'] = calc.halo.rb1
        hdr['magneticData']['magneticScale2'] = calc.halo.rb2
    elif(calc.phys.b_flag == "exp"):
        if calc.phys.qb == 0.0:
            hdr['magneticData']['magneticScale'] = calc.halo.r_stellar_half_light
        else:
            hdr['magneticData']['magneticScale'] = calc.phys.qb
    elif(calc.phys.b_flag == "m31"):
        hdr['magneticData']['magneticScale'] = calc.phys.qb
    elif(calc.phys.b_flag == "m31exp"):
        hdr['magneticData']['magneticScale'] = calc.phys.qb 
    if(calc.phys.diff == 0):
        hdr['diffData']['diffFlag'] = 0
    else:
        hdr['diffData']['diffFlag'] = 1
        hdr['diffData']['diffSCale' ] = calc.phys.lc*1e-3
        hdr['diffData']['diffIndex'] = calc.phys.delta
        hdr['diffData']['diffConstant'] = calc.phys.d0
    return hdr
    

def fitsHeaderGB(calc,hdr):
    hdr['GASPROF'] = calc.phys.ne_model
    hdr['GASN0'] = calc.phys.ne0
    hdr['GAVG'] = calc.halo.neav
    if(calc.phys.ne_model == "powerlaw" or calc.phys.ne_model == "pl" or calc.phys.ne_model == "king"):
        if calc.phys.lb is None:
            hdr['GSCALE'] = calc.halo.rcore
        else:
            hdr['GSCALE'] = calc.phys.lb
        hdr['GINDEX'] = -1*calc.phys.qe
    elif(calc.phys.ne_model == "exp"):
        if calc.halo.r_stellar_half_light is None:
            hdr['GSCALE'] = calc.halo.phys.lb
        else:
            hdr['GSCALE'] = calc.halo.r_stellar_half_light
    hdr['BPROF'] = calc.phys.b_flag
    hdr['B0'] = calc.phys.b0
    hdr['BAV'] = calc.halo.bav
    if(calc.phys.b_flag == "powerlaw" or calc.phys.b_flag == "pl"):
        hdr['BINDEX'] = -1*calc.phys.qb*calc.phys.qe
    elif(calc.phys.b_flag == "follow_ne"):
        hdr['BINDEX'] = -1*calc.phys.qb*calc.phys.qe
    elif calc.phys.b_flag == "sc2006":
        hdr['BSCALE1'] = calc.halo.rb1
        hdr['BSCALE2'] = calc.halo.rb2
    elif(calc.phys.b_flag == "exp"):
        if calc.phys.qb == 0.0:
            hdr['BSCALE'] = calc.halo.r_stellar_half_light
        else:
            hdr['BSCALE'] = calc.phys.qb
    elif(calc.phys.b_flag == "m31"):
        hdr['BSCALE'] = calc.phys.qb
    elif(calc.phys.b_flag == "m31exp"):
        hdr['BSCALE'] = calc.phys.qb 
    if(calc.phys.diff == 0):
        hdr['DIFF'] = 0
    else:
        hdr['DIFF'] = 1
        hdr['TSCALE' ] = calc.phys.lc*1e-3
        hdr['TDELTA'] = calc.phys.delta
        hdr['D0'] = calc.phys.d0
    return hdr