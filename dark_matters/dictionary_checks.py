import os,yaml
import numpy as np
from astropy import constants
from genericpath import isdir

from .input import getSpectralData
from .output import fatal_error,warning
from .astro_cosmo import astrophysics,cosmology

def checkCosmology(cosmoDict):
    """
    Checks the properties of a cosmology dictionary

    Arguments
    ---------------------------
    cosmoDict : dictionary
        Cosmology information

    Returns
    ---------------------------
    cosmoDict : dictionary
        Cosmology information in compliance with DarkMatters requirements, code will exit if this cannot be achieved
    """
    if not type(cosmoDict) is dict:
        fatal_error("control.checkCosmology() must be passed a dictionary as its argument")
    if not 'omega_m' in cosmoDict.keys() and not 'omega_l' in cosmoDict.keys():
        cosmoDict['omega_m'] = 0.3089
        cosmoDict['omega_l'] = 1 - cosmoDict['omega_m']
    elif not 'omega_m' in cosmoDict.keys():
        cosmoDict['omega_m'] = 1 - cosmoDict['omega_l']
    elif not 'omega_l' in cosmoDict.keys():
        cosmoDict['omega_l'] = 1 - cosmoDict['omega_m']
    if not 'cvirMode' in cosmoDict.keys():
        cosmoDict['cvirMode'] = 'p12'
    cvir_modes = ['p12','munoz_2010','bullock_2001','cpu_2006']
    if not cosmoDict['cvirMode'] in cvir_modes:
        fatal_error(f"cosmoData['cvirMode'] = {cosmoDict['cvirMode']} is not valid, use one of {cvir_modes}")
    if not 'h' in cosmoDict.keys():
        cosmoDict['h'] = 0.6774
    return cosmoDict

def checkMagnetic(magDict):
    """
    Checks the properties of a magnetic field dictionary

    Arguments
    ---------------------------
    magDict : dictionary
        Magnetic field information

    Returns
    ---------------------------
    magDict : dictionary
        Magnetic Field information in compliance with DarkMatters requirements, code will exit if this cannot be achieved
    """
    if not type(magDict) is dict:
        fatal_error("control.checkMagnetic() must be passed a dictionary as its argument")
    inFile = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"config/magFieldProfiles.yaml"),"r")
    profileDict = yaml.load(inFile,Loader=yaml.SafeLoader)
    inFile.close()
    if 'magFieldFunc' in magDict.keys() and magDict['magFuncLock']: #No functionality implemented yet
        magDict['magProfile'] = "custom"
        return magDict
    if not 'magProfile' in magDict.keys():
        magDict['magProfile'] = "flat"
    if not magDict['magProfile'] in profileDict.keys():
        fatal_error(f"magData variable magProfile is required to be one of {profileDict.keys()}")
    needVars = profileDict[magDict['magProfile']]
    for var in needVars:
        if not var in magDict.keys():
            fatal_error(f"magData variable {var} required for magnetic field profile {magDict['magProfile']}")
        if not np.isscalar(magDict[var]):
            fatal_error(f"magData property {var} must be a scalar")
    if not magDict['magFuncLock']:
        magDict['magFieldFunc'] = astrophysics.magneticFieldBuilder(magDict)
    if magDict['magFieldFunc'] is None:
        fatal_error(f"No magFieldFunc recipe for profile {magDict['magProfile']} found in astrophysics.magneticFieldBuilder()")
    return magDict

def checkHalo(haloDict,cosmoDict,minimal=False):
    """
    Checks the properties of a halo dictionary

    Arguments
    ---------------------------
    haloDict : dictionary
        Halo properties
    cosmoDict : dictionary
        Cosmology information, must have been checked via checkCosmology

    Returns
    ---------------------------
    haloDict : dictionary
        Halo information in compliance with DarkMatters requirements, code will exit if this cannot be achieved
    """
    if not type(haloDict) is dict:
        fatal_error("control.checkHalo() must be passed a dictionary as its argument")
    if ((not 'haloZ' in haloDict.keys()) or haloDict['haloZ'] == 0.0) and not 'haloDistance' in haloDict.keys():
        fatal_error("Either haloDistance must be specified or haloZ must be non-zero")
    elif not 'haloZ' in haloDict.keys():
        haloDict['haloZ'] = 0.0
    elif not 'haloDistance' in haloDict.keys():
        haloDict['haloDistance'] = cosmology.dist_luminosity(haloDict['haloZ'],cosmoDict)
    if minimal:
        return haloDict
    inFile = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"config/haloDensityProfiles.yaml"),"r")
    haloParams = yaml.load(inFile,Loader=yaml.SafeLoader)
    inFile.close()
    def rhoNorm(haloDict,cosmoDict):
        if (not "haloNorm" in haloDict.keys()) and ("haloNormRelative" in haloDict.keys()):
            haloDict['haloNorm'] = haloDict['haloNormRelative']*cosmology.rho_crit(haloDict['haloZ'],cosmoDict)
        elif ("haloNorm" in haloDict.keys()) and (not "haloNormRelative" in haloDict.keys()):
            haloDict['haloNormRelative'] = haloDict['haloNorm']/cosmology.rho_crit(haloDict['haloZ'],cosmoDict)
        else:
            haloDict['haloNorm'] = haloDict['haloMvir']/astrophysics.rhoVirialInt(haloDict)
            haloDict['haloNormRelative'] = haloDict['haloNorm']/cosmology.rho_crit(haloDict['haloZ'],cosmoDict)
        return haloDict

    if not 'haloWeights' in haloDict.keys():
        haloDict['haloWeights'] = "rho"
    if not 'haloProfile' in haloDict.keys():
        fatal_error("halo variable haloProfile is required for non J/D-factor calculations")
    
    varSet1 = ["haloNorm","haloMvir","haloRvir","haloNormRelative"]
    varSet2 = ["haloCvir","haloScale"]
    if ((not len(set(varSet1).intersection(haloDict.keys())) > 0) or (not len(set(varSet2).intersection(haloDict.keys())) > 0)) and not ("haloRvir" in haloDict.keys() or "haloMvir" in haloDict.keys()):
        fatal_error(f"Halo specification requires 1 halo variable from {varSet1} and 1 from {varSet2}")
    else:
        if haloDict['haloProfile'] not in haloParams.keys():
            fatal_error(f"Halo specification requires haloProfile from: {haloParams.keys()}")
        else:
            for x in haloParams[haloDict['haloProfile']]:
                if not x == "none":
                    if not x in haloDict.keys():
                        fatal_error(f"haloProfile {haloDict['haloProfile']} requires property {x} be set")
        if haloDict["haloProfile"] == "burkert":
            #rescale to reflect where dlnrho/dlnr = -2 (required as cvir = rvir/r_{-2})
            #isothermal, nfw, einasto all have rs = r_{-2}
            scaleMod = 1.5214
        elif haloDict["haloProfile"] == "gnfw":
            scaleMod = 2.0 - haloDict['haloIndex']
        else:
            scaleMod = 1.0
        rsInfo = "haloScale" in haloDict.keys() 
        rhoInfo = "haloNorm" in haloDict.keys() or "haloNormRelative" in haloDict.keys()
        rvirInfo = "haloRvir" in haloDict.keys() or "haloMvir" in haloDict.keys()
        if  rsInfo and rhoInfo:
            if not "haloNormRelative" in haloDict.keys():
                haloDict['haloNormRelative'] = haloDict['haloNorm']/cosmology.rho_crit(haloDict['haloZ'],cosmoDict)
            elif not "haloNorm" in haloDict.keys():
                haloDict['haloNorm'] = haloDict['haloNormRelative']*cosmology.rho_crit(haloDict['haloZ'],cosmoDict)
            if (not "haloMvir" in haloDict.keys()) and ("haloRvir" in haloDict.keys()):
                haloDict['haloMvir'] = haloDict['haloNorm']*astrophysics.rho_volume_int(haloDict)
            elif ("haloMvir" in haloDict.keys()) and (not "haloRvir" in haloDict.keys()):
                haloDict['haloRvir'] = cosmology.rvirFromMvir(haloDict['haloMvir'],haloDict['haloZ'],cosmoDict)
            elif (not "haloMvir" in haloDict.keys()) and (not "haloRvir" in haloDict.keys()):
                haloDict['haloRvir']= astrophysics.rvirFromRho(haloDict,cosmoDict)
                haloDict['haloMvir'] = cosmology.mvirFromRvir(haloDict['haloRvir'],haloDict['haloZ'],cosmoDict)
            if not "haloCvir" in haloDict.keys():
                haloDict['haloCvir'] = haloDict['haloRvir']/haloDict['haloScale']/scaleMod
        elif rvirInfo and rsInfo:
            if not 'haloRvir' in haloDict.keys():
                haloDict['haloRvir'] = cosmology.rvirFromMvir(haloDict['haloMvir'],haloDict['haloZ'],cosmoDict)
            if not 'haloCvir' in haloDict.keys():
                haloDict['haloCvir'] = haloDict['haloRvir']/haloDict['haloScale']/scaleMod
            if not 'haloMvir' in haloDict.keys():
                haloDict['haloMvir'] = cosmology.mvirFromRvir(haloDict['haloRvir'],haloDict['haloZ'],cosmoDict)
            else:
                if not 'haloRvir' in haloDict.keys():
                    haloDict['haloRvir'] = cosmology.rvirFromMvir(haloDict['haloMvir'],haloDict['haloZ'],cosmoDict)
                if not 'haloCvir' in haloDict.keys():
                    haloDict['haloCvir'] = haloDict['haloRvir']/haloDict['haloScale']/scaleMod
            haloDict = rhoNorm(haloDict,cosmoDict)
        elif rvirInfo and 'haloCvir' in haloDict.keys():
            if not 'haloMvir' in haloDict.keys():
                haloDict['haloMvir'] = cosmology.mvirFromRvir(haloDict['haloRvir'],haloDict['haloZ'],cosmoDict)
            if not 'haloRvir' in haloDict.keys():
                haloDict['haloRvir'] = cosmology.rvirFromMvir(haloDict['haloMvir'],haloDict['haloZ'],cosmoDict)
            if not 'haloScale' in haloDict.keys():
                haloDict['haloScale'] = haloDict['haloRvir']/haloDict['haloCvir']/scaleMod
            haloDict = rhoNorm(haloDict,cosmoDict)
        elif rvirInfo:
            if not 'haloMvir' in haloDict.keys():
                haloDict['haloMvir'] = cosmology.mvirFromRvir(haloDict['haloRvir'],haloDict['haloZ'],cosmoDict)
            if not 'haloRvir' in haloDict.keys():
                haloDict['haloRvir'] = cosmology.rvirFromMvir(haloDict['haloMvir'],haloDict['haloZ'],cosmoDict)
            haloDict['haloCvir'] = cosmology.cvir(haloDict['haloMvir'],haloDict['haloZ'],cosmoDict)
            haloDict['haloScale'] = haloDict['haloRvir']/haloDict['haloCvir']/scaleMod
            haloDict = rhoNorm(haloDict,cosmoDict)
        else:
            fatal_error(f"haloData is underspecified by {haloDict}")
    haloDict['haloDensityFunc'] = astrophysics.haloDensityBuilder(haloDict)
    if not "greenAveragingScale" in haloDict.keys():
        haloDict["greenAveragingScale"] = haloDict['haloScale']
    if haloDict['haloDensityFunc'] is None:
        fatal_error(f"No haloDensityFunc recipe for profile {haloDict['haloProfile']} found in astrophysics.haloDensityBuilder()")
    return haloDict

def checkGas(gasDict):
    """
    Checks the properties of a gas dictionary

    Arguments
    ---------------------------
    gasDict : dictionary
        Gas properties

    Returns
    ---------------------------
    gasDict : dictionary
        Gas information in compliance with DarkMatters requirements, code will exit if this cannot be achieved
    """
    if not type(gasDict) is dict:
        fatal_error("control.checkGas() must be passed a dictionary as its argument")
    inFile = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"config/gasDensityProfiles.yaml"),"r")
    gasParams = yaml.load(inFile,Loader=yaml.SafeLoader)
    inFile.close()
    if not 'gasProfile' in gasDict.keys():
        gasDict['gasProfile'] = "flat"
    else:
        needVars = gasParams[gasDict['gasProfile']]
        for var in needVars:
            if not var in gasDict.keys():
                print(f"gasData variable {var} is required for magnetic field profile {gasDict['gasProfile']}")
                fatal_error("gasData underspecified")
    gasDict['gasDensityFunc'] = astrophysics.gasDensityBuilder(gasDict)
    if gasDict['gasDensityFunc'] is None:
        fatal_error(f"No gasDensityFunc recipe for profile {gasDict['gasProfile']} found in astrophysics.gasDensityBuilder()")
    return gasDict   

def checkCalculation(calcDict):
    """
    Checks the properties of a calculation dictionary

    Arguments
    ---------------------------
    calcDict : dictionary
        Calculation information

    Returns
    ---------------------------
    calcDict : dictionary 
        Calculation information in compliance with DarkMatters requirements, code will exit if this cannot be achieved
    """
    if not type(calcDict) is dict:
        fatal_error("control.checkCalculation() must be passed a dictionary as its argument")
    calcParams = {'allElectronModes':["adi-python","green-python","green-c"],'allModes':["jflux","flux","sb"],"allFreqs":["radio","all","gamma","pgamma","sgamma","neutrinos_e","neutrinos_mu","neutrinos_tau"]}
    if not 'mWIMP' in calcDict.keys():
        fatal_error("calcDict requires the variable mWIMP be set")
    if not 'calcMode' in calcDict.keys() or (not calcDict['calcMode'] in calcParams['allModes']):
        fatal_error(f"calcDict requires the variable calcMode with options: {calcParams['allModes']}")
    if not 'freqMode' in calcDict.keys() or (not calcDict['freqMode'] in calcParams['allFreqs']):
        fatal_error(f"calcDict requires the variable freqMode with options: {calcParams['allFreqs']}")
    if not 'electronMode' in calcDict.keys():
        calcDict['electronMode'] = "adi-python"  
    elif calcDict['electronMode'] not in calcParams['allElectronModes']:
        fatal_error(f"electronMode can only take the values: green-python, green-c, or adi-python. Your value of {calcDict['electronMode']} is invalid")
    if not 'outCGS' in calcDict.keys():
        calcDict['outCGS'] = False
    if not 'fSampleValues' in calcDict.keys(): 
        if not 'fSampleLimits' in calcDict.keys():
            fatal_error("calcDict requires one of the following variables: fSampleLimits, giving the minimum and maximum frequencies to be studied OR fSampleValues, an array of explicitly sampled frequencies")
        if not 'fSampleNum' in calcDict.keys():
            calcDict['fSampleNum'] = int((np.log10(calcDict['fSampleLimits'][1]) - np.log10(calcDict['fSampleLimits'][0]))/5)
        if not 'fSampleSpacing' in calcDict.keys():
            calcDict['fSampleSpacing'] = "log"
        if calcDict['fSampleSpacing'] == "lin":
            calcDict['fSampleValues'] = np.linspace(calcDict['fSampleLimits'][0],calcDict['fSampleLimits'][1],num=calcDict['fSampleNum'])
        else:
            calcDict['fSampleValues'] = np.logspace(np.log10(calcDict['fSampleLimits'][0]),np.log10(calcDict['fSampleLimits'][1]),num=calcDict['fSampleNum'])
    else:
        calcDict['fSampleNum'] = len(calcDict['fSampleValues'])
        calcDict['fSampleLimits'] = [calcDict['fSampleValues'][0],calcDict['fSampleValues'][-1]]
        calcDict['fSampleSpacing'] = "custom"

    if not 'eSampleNum' in calcDict.keys():
        if 'green' in calcDict['electronMode']:
            calcDict['eSampleNum'] = 50
        else:
            calcDict['eSampleNum'] = 80
    if not 'rSampleNum' in calcDict.keys():
        if 'green' in calcDict['electronMode']:
            calcDict['rSampleNum'] = 50
        else:
            calcDict['rSampleNum'] = 80
    if not 'log10RSampleMinFactor' in calcDict.keys():
        calcDict['log10RSampleMinFactor'] = -2
    if not 'eSampleMin' in calcDict.keys():
        calcDict['eSampleMin'] = (constants.m_e*constants.c**2).to("GeV").value #GeV

    if calcDict['calcMode'] in ["flux"]:
        if (not 'calcRmaxIntegrate' in calcDict.keys()) and (not 'calcAngmaxIntegrate' in calcDict.keys()):
            fatal_error(f"calcDict requires one of the variables calcRmaxIntegrate or calcAngmaxIntegrate for the selected mode: {calcDict['calcMode']}")
        elif ('calcRmaxIntegrate' in calcDict.keys()) and ('calcAngmaxIntegrate' in calcDict.keys()):
            fatal_error(f"calcDict requires ONLY one of the variables calcRmaxIntegrate or calcAngmaxIntegrate for the selected mode: {calcDict['calcMode']}")

    if not calcDict['calcMode'] == "jflux":
        if "green" in calcDict['electronMode']: 
            if not 'threadNumber' in calcDict.keys():
                calcDict['threadNumber'] = 4
            if not "imageNumber" in calcDict.keys():
                calcDict['imageNumber'] = 30
            if (not 'electronExecFile' in calcDict.keys()):
                calcDict['electronExecFile'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),"emissions/electron.x")
            if (not 'rGreenSampleNum' in calcDict.keys()):
                calcDict['rGreenSampleNum'] = 61
            if calcDict['rGreenSampleNum'] < 61:
                fatal_error("rGreenSampleNum cannot be set below 61 without incurring errors")
            if (not 'eGreenSampleNum' in calcDict.keys()):
                calcDict['eGreenSampleNum'] = 401
            if calcDict['eGreenSampleNum'] < 201:
                fatal_error("eGreenSampleNum cannot be set below 201 without incurring substantial errors, recommended value is 401")
            if calcDict['eGreenSampleNum'] < 401:
                warning(f"eGreenSampleNum recommended value is 401 to minimize errors, you are curently using {calcDict['eGreenSampleNum']}")
            if (calcDict['rGreenSampleNum']-1)%4 != 0:
                fatal_error(f"rGreenSampleNum - 1 must be divisible by 4, you provided {calcDict['rGreenSampleNum']}")
            if (calcDict['eGreenSampleNum']-1)%4 != 0:
                fatal_error(f"eGreenSampleNum - 1 must be divisible by 4, you provided {calcDict['eGreenSampleNum']}")
        elif calcDict['electronMode'] == "adi-python":
            if not "adiDeltaTi" in calcDict.keys():
                calcDict['adiDeltaTi'] = 1e9 
            if not "adiDeltaTReduction" in calcDict.keys():
                calcDict['adiDeltaTReduction'] = 0.5 
            if not "adiMaxSteps" in calcDict.keys():
                calcDict['adiMaxSteps'] = 100
            if not "adiDeltaTConstant" in calcDict.keys():
                calcDict['adiDeltaTConstant'] = False 
            if not "adiBenchMarkMode" in calcDict.keys():
                calcDict['adiBenchMarkMode'] = False  
            if not 'adiDeltaTMin' in calcDict.keys():
                calcDict['adiDeltaTMin'] = 1e1  
    else:
        if not calcDict['freqMode'] in ["pgamma","neutrinos_mu","neutrinos_e","neutrinos_tau"]:
            fatal_error("calcData freqMode parameter can only be pgamma, or neutrinos_x (x= e, mu, or tau) for calcMode jflux")
    greenOnlyParams = ["rGreenSampleNum","eGreenSampleNum","threadNumber","imageNumber"]
    adiOnlyParams = ["adiDeltaTReduction","adiDeltaTi","adiMaxSteps","adiDeltaTConstant","adiBenchMarkMode","adiDeltaTMin"]
    if "green" in calcDict['electronMode']:
        for p in adiOnlyParams:
            if p in calcDict.keys():
                calcDict.pop(p)
    else:
        for p in greenOnlyParams:
            if p in calcDict.keys():
                calcDict.pop(p)
    return calcDict 

def checkParticles(partDict,calcDict):
    """
    Checks the properties of a particle physics dictionary

    Arguments
    ---------------------------
    partDict : dictionary
        Particle physics information
    calcDict : dictionary
        Calculation information, must have been checkCalculation'd first

    Returns
    ---------------------------
    partDict : dictionary 
        Particle physics in compliance with DarkMatters requirements, code will exit if this cannot be achieved
    """
    if not type(calcDict) is dict or not type(partDict) is dict:
        fatal_error("control.checkParticles() must be passed a dictionaries as its argument")
    if not 'partModel' in partDict.keys():
        fatal_error("partDict requires a partModel value")
    if not 'emModel' in partDict.keys():
        partDict['emModel'] = "annihilation"
    elif not partDict['emModel'] in ["annihilation","decay"]:
        fatal_error("emModel must be set to either annihilation or decay")
    if not 'decayInput' in partDict.keys():
        partDict['decayInput'] = False
    if not 'spectrumDirectory' in partDict.keys():
        partDict['spectrumDirectory'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),"particle_physics")
    if not isdir(partDict['spectrumDirectory']):
        warning(f"partData['spectrumDirectory'] = {partDict['spectrumDirectory']} is not a valid folder, using default instead")
        partDict['spectrumDirectory'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),"particle_physics")
    specSet = []
    if "neutrinos" in calcDict['freqMode']:
        specSet.append(calcDict['freqMode'])
    if calcDict['freqMode'] in ["gamma","pgamma","sgamma","all"]:
        specSet.append("gammas")
    if calcDict['freqMode'] in ['sgamma',"radio","all","gamma"]:
        specSet.append("positrons")
    partDict['dNdxInterp'] = getSpectralData(partDict['spectrumDirectory'],partDict['partModel'],specSet,mode=partDict["emModel"])
    if 'crossSection' in partDict.keys() and 'decayRate' in partDict.keys():
        fatal_error("You cannot have both a crossSection and decayRate set for the particle physics")
    elif not 'crossSection' in partDict.keys() and not 'decayRate' in partDict.keys():
        if partDict['emModel'] == "annihilation":
            partDict['crossSection'] = 1e-26
        else:
            partDict['decayRate'] = 1e-26
    return partDict

def checkDiffusion(diffDict):
    """
    Checks the properties of a diffusion dictionary

    Arguments
    ---------------------------
    diffDict : dictionary
        Diffusion information

    Returns
    ---------------------------
    diffDict : dictionary
        Diffusion information in compliance with DarkMatters requirements, code will exit if this cannot be achieved
    """
    if not type(diffDict) is dict:
        fatal_error("control.checkDiffusion() must be passed a dictionary as its argument")
    if not 'lossOnly' in diffDict.keys():
        diffDict['lossOnly'] = False
    if not 'photonDensity' in diffDict.keys():
        diffDict['photonDensity'] = 0.0
    if not 'photonTemp' in diffDict.keys():
        diffDict['photonTemp'] = 2.7255
    if not 'diffRmax' in diffDict.keys():
        diffDict['diffRmax'] = "2*Rvir"
    if diffDict['lossOnly']:
        diffDict['diffConstant'] = 0.0
    else:
        if not 'diffConstant' in diffDict.keys():
            diffDict['diffConstant'] = 3e28
        if not 'diffIndex' in diffDict.keys():
            diffDict['diffIndex'] = 1.0/3
    return diffDict