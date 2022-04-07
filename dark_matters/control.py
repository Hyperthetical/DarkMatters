from genericpath import isfile
import numpy as np
from os.path import join
import os,yaml
from scipy.integrate import simps
from scipy.optimize import newton
from astropy import constants
import sympy
from sympy.utilities.lambdify import lambdify
from .input import getSpectralData
from .output import fatal_error,calcWrite,wimpWrite


from .astro_cosmo import astrophysics,cosmology
from .emissions import electron,fluxes,emissivity,cn_electron

def getIndex(set,val):
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
    if 'magFieldFunc' in magDict.keys() and magDict['magFuncLock']:
        magDict['magProfile'] = "custom"
        return magDict
    if not 'magProfile' in magDict.keys():
        magDict['magProfile'] = "flat"
    if not magDict['magProfile'] in profileDict.keys():
        fatal_error("magData variable magProfile is required to be one of {}".format(profileDict.keys()))
    needVars = profileDict[magDict['magProfile']]
    for var in needVars:
        if not var in magDict.keys():
            fatal_error("magData variable {} required for magnetic field profile {}".format(var,magDict['magProfile']))
        if not np.isscalar(magDict[var]):
            fatal_error("magData property {} must be a scalar".format(var))
    if not magDict['magFuncLock']:
        magDict['magFieldFunc'] = astrophysics.magneticFieldBuilder(magDict)
    if magDict['magFieldFunc'] is None:
        fatal_error("No magFieldFunc recipe for profile {} found in astrophysics.magneticFieldBuilder()".format(magDict['magProfile']))
    return magDict

def checkHalo(haloDict,cosmoDict):
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
        fatal_error("halo variable {} is required for non J/D-factor calculations".format('haloProfile'))
    
    if not 'haloZ' in haloDict.keys():
        haloDict['haloZ'] = 0.0
        if not 'haloDistance' in haloDict.keys():
            fatal_error("either haloZ or haloDistance must be specified in haloData")
    elif not 'haloDistance' in haloDict.keys():
        haloDict['haloDistance'] = cosmology.dist_luminosity(haloDict['haloZ'],cosmoDict)
    varSet1 = ["haloNorm","haloMvir","haloRvir","haloNormRelative"]
    varSet2 = ["haloCvir","haloScale"]
    if ((not len(set(varSet1).intersection(haloDict.keys())) > 0) or (not len(set(varSet2).intersection(haloDict.keys())) > 0)) and not ("haloRvir" in haloDict.keys() or "haloMvir" in haloDict.keys()):
        fatal_error("Halo specification requires 1 halo variable from {} and 1 from {}".format(varSet1,varSet2))
    else:
        if haloDict['haloProfile'] not in haloParams.keys():
            fatal_error("Halo specification requires haloProfile from: {}".format(haloParams.keys()))
        else:
            for x in haloParams[haloDict['haloProfile']]:
                if not x == "none":
                    if not x in haloDict.keys():
                        fatal_error("haloProfile {} requires property {} be set".format(haloDict['haloProfile'],x))
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
            haloDict['haloCvir'] = cosmology.cvir_p12_param(haloDict['haloMvir'],haloDict['haloZ'],cosmoDict)
            haloDict['haloScale'] = haloDict['haloRvir']/haloDict['haloCvir']/scaleMod
            haloDict = rhoNorm(haloDict,cosmoDict)
        else:
            fatal_error("haloData is underspecified by {}".format(haloDict))
    haloDict['haloDensityFunc'] = astrophysics.haloDensityBuilder(haloDict)
    if haloDict['haloDensityFunc'] is None:
        fatal_error("No haloDensityFunc recipe for profile {} found in astrophysics.haloDensityBuilder()".format(haloDict['haloProfile']))
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
                print("gasData variable {} is required for magnetic field profile {}".format(var,gasDict['gasProfile']))
                fatal_error("gasData underspecified")
    gasDict['gasDensityFunc'] = astrophysics.gasDensityBuilder(gasDict)
    if gasDict['gasDensityFunc'] is None:
        fatal_error("No gasDensityFunc recipe for profile {} found in astrophysics.gasDensityBuilder()".format(gasDict['gasProfile']))
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
    calcParams = {'allElectronModes':["crank-python","green-python","green-c"],'allModes':["jflux","flux","sb"],"allFreqs":["radio","all","gamma","pgamma","sgamma","neutrinos_e","neutrinos_mu","neutrinos_tau"]}
    if not 'mWIMP' in calcDict.keys():
        fatal_error("calcDict requires the variable {} be set".format('mWIMP'))
    else:
        for i in range(len(calcDict['mWIMP'])):
            calcDict['results']['electronData'].append(None)
            calcDict['results']['radioEmData'].append(None)
            calcDict['results']['primaryEmData'].append(None)
            calcDict['results']['secondaryEmData'].append(None)
            calcDict['results']['neutrinoEmData'].append(None)
            calcDict['results']['finalData'].append(None)
    if not 'calcMode' in calcDict.keys() or (not calcDict['calcMode'] in calcParams['allModes']):
        fatal_error("calcDict requires the variable {} with options: {}".format('calcMode',calcParams['allModes']))
    if not 'freqMode' in calcDict.keys() or (not calcDict['freqMode'] in calcParams['allFreqs']):
        fatal_error("calcDict requires the variable {} with options: {}".format('freqMode',calcParams['allFreqs']))
    if not 'electronMode' in calcDict.keys():
        calcDict['electronMode'] = "green-python"  
    if calcDict['electronMode'] == "green-c":  
        if not 'threadNumber' in calcDict.keys():
            calcDict['threadNumber'] = 4
    elif calcDict['electronMode'] == "crank-python":
        if not "crankDeltaTi" in calcDict.keys():
            calcDict['crankDeltaTi'] = 1e9 
        if not "crankDeltaTReduction" in calcDict.keys():
            calcDict['crankDeltaTReduction'] = 0.5 
        if not "crankMaxSteps" in calcDict.keys():
            calcDict['crankMaxSteps'] = 100
        if not "crankDeltaTConstant" in calcDict.keys():
            calcDict['crankDeltaTConstant'] = False 
        if not "crankBenchMarkMode" in calcDict.keys():
            calcDict['crankBenchMarkMode'] = False  
    elif calcDict['electronMode'] not in calcParams['allElectronModes']:
        fatal_error("electronMode can only take the values: green-python, green-c, or crank-python. Your value of {} is invalid.".format(calcDict['electronMode'])) 
    if not 'fSampleValues' in calcDict.keys(): 
        if not 'fSampleLimits' in calcDict.keys():
            fatal_error("calcDict requires the variable {}, giving the minimum and maximum frequencies to be studied".format('fSampleLimits'))
        if not 'fSampleNum' in calcDict.keys():
            calcDict['fSampleNum'] = 40
        if not 'fSampleSpacing' in calcDict.keys():
            calcDict['fSampleSpacing'] = "lin"
        if calcDict['fSampleSpacing'] == "lin":
            calcDict['fSampleValues'] = np.linspace(calcDict['fSampleLimits'][0],calcDict['fSampleLimits'][1],num=calcDict['fSampleNum'])
        else:
            calcDict['fSampleValues'] = np.logspace(np.log10(calcDict['fSampleLimits'][0]),np.log10(calcDict['fSampleLimits'][1]),num=calcDict['fSampleNum'])
    else:
        calcDict['fSampleNum'] = len(calcDict['fSampleValues'])
        calcDict['fSampleLimits'] = [calcDict['fSampleValues'][0],calcDict['fSampleValues'][-1]]
        calcDict['fSampleSpacing'] = "custom"

    if not 'eSampleNum' in calcDict.keys():
        calcDict['eSampleNum'] = 71
    elif calcDict['eSampleNum'] < 71:
        fatal_error("eSampleNum cannot be set below 71 without incurring errors")
    if not 'log10ESampleMinFactor' in calcDict.keys():
        calcDict['log10ESampleMinFactor'] = -9

    if calcDict['calcMode'] == "flux":
        if (not 'calcRmax' in calcDict.keys()) and (not 'calcAngmax' in calcDict.keys()):
            fatal_error("calcDict requires one of the variables {} or {} for the selected mode: {}".format('calcRmax','calcAngmax',calcDict['calcMode']))
        elif ('calcRmax' in calcDict.keys()) and ('calcAngmax' in calcDict.keys()):
            fatal_error("calcDict requires ONLY one of the variables {} or {} for the selected mode: {}".format('calcRmax','calcAngmax',calcDict['calcMode']))
    if not calcDict['calcMode'] == "jflux":
        if not 'rSampleNum' in calcDict.keys():
            calcDict['rSampleNum'] = 61
        if not 'rGreenSampleNum' in calcDict.keys():
            calcDict['rGreenSampleNum'] = 121
        if not 'log10RSampleMinFactor' in calcDict.keys():
            calcDict['log10RSampleMinFactor'] = -5
    else:
        if not calcDict['freqMode'] in ["pgamma","neutrinos_mu","neutrinos_e","neutrinos_tau"]:
            fatal_error("calcData freqMode parameter can only be pgamma, or neutrinos_x (x= e, mu, or tau) for calcMode jflux")
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
    if not 'spectrumDirectory' in partDict.keys():
        partDict['spectrumDirectory'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),"particle_physics")
    try:
        open(partDict['spectrumDirectory'],"r")
    except:
        partDict['spectrumDirectory'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),"particle_physics")
    specSet = []
    if "neutrinos" in calcDict['freqMode']:
        specSet.append(calcDict['freqMode'])
    if calcDict['freqMode'] in ["gamma","pgamma","sgamma","all"]:
        specSet.append("gammas")
    if calcDict['freqMode'] in ['sgamma',"radio","all","gamma"]:
        specSet.append("positrons")
    partDict['dNdxInterp'] = getSpectralData(partDict['spectrumDirectory'],partDict['partModel'],specSet,mode=partDict["emModel"])
    #check spectrum files and create 2D interpolation function
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
        fatal_error("control.checkDiffusion() must be passed a dictionaries as its argument")
    if not 'ISRF' in diffDict.keys():
        diffDict['ISRF'] = False
    if not 'lossOnly' in diffDict.keys():
        diffDict['lossOnly'] = True
        return diffDict
    if diffDict['lossOnly']:
        return diffDict
    else:
        if not 'coherenceScale' in diffDict.keys():
            fatal_error("diffDict requires the variable {} when lossOnly = False".format('coherenceScale'))
        if not 'diffConstant' in diffDict.keys():
            diffDict['diffConstant'] = 3e28
        if not 'diffIndex' in diffDict.keys():
            diffDict['diffIndex'] = 5.0/3
        return diffDict


def takeSamples(xmin,xmax,nx,spacing="log"):
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
        return np.logspace(np.log10(xmin),np.log10(xmax),num=nx)
    else:
        return np.linspace(xmin,xmax,num=nx)

def physical_averages(rmax,mode_exp,calcData,haloData,magData,gasData):
    """
    Computes averages for magnetic field and halo data within radius rmax

    Arguments
    ---------------------------
    rmax : float
        Integration limit
    mode_exp  : float
        Flag, 1 for decay or 2 for annihilation
    calcData : dictionary
        Calculation properties
    haloData : dictionary
        Halo properties
    magData : dictionary
        Magnetic field
    gasData : dictionary
        Gas distribution

    Returns
    ---------------------------
    b_av, ne_av : float,float
        Volume averages
    """
    def weightedVolAvg(y,r,w=None):
        if w is None:
            w = np.ones_like(r)
        return simps(y*w*r**2,r)/simps(w*r**2,r)
    rSet = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rmax,100)
    if haloData['haloWeights'] == "rho":
        weights = haloData['haloDensityFunc'](rSet)**mode_exp #the average is weighted
    else:
        weights = np.ones_like(rSet)
    return weightedVolAvg(magData['magFieldFunc'](rSet),rSet,weights),weightedVolAvg(gasData['gasDensityFunc'](rSet),rSet,weights)


def calcElectrons(mx,calcData,haloData,partData,magData,gasData,diffData):
    """
    Computes equilibrium electron distributions from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calcData : dictionary
        Calculation properties
    haloData : dictionary
        Halo properties
    partData : dictionary
        Particle physics
    magData : dictionary
        Magnetic field
    gasData : dictionary
        Das distribution
    diffData : dictionary
        Diffusion properties

    Returns
    ---------------------------
    calcData : dictionary
        Calculation information with electron distribution in calcData['results']['electronData']
    """
    mIndex = getIndex(calcData['mWIMP'],mx)
    if partData['emModel'] == "annihilation":
        mode_exp = 2.0
    else:
        mode_exp = 1.0
    mxEff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mxEff = mx
    if diffData["ISRF"]:
        ISRF = 1
    else:
        ISRF = 0
    if diffData['lossOnly']:
        diff = 0
        lc = 0
        delta = 0
        d0 = 3.1e28
        if calcData['rSampleNum'] < 51:
            fatal_error("You have specified rSampleNum = {}, this will yield inaccurate electron density results.\n Set rSampleNum > 50".format(calcData['rSampleNum'])) 
    else:
        diff = 1
        lc = diffData['coherenceScale']*1e3 #kpc
        delta = diffData['diffIndex']
        d0 = diffData['diffConstant']
        if calcData['rSampleNum'] < 51:
            fatal_error("You have specified rSampleNum = {}, this will yield inaccurate electron density results with lossOnly = False.\n Set rSampleNum > 70".format(calcData['rSampleNum'])) 
        if calcData['rGreenSampleNum'] < 101:
            fatal_error("You have specified rGreenSampleNum = {}, this will yield inaccurate electron density results with lossOnly = False.\n Set rGreenSampleNum > 200".format(calcData['rGreenSampleNum'])) 
    if 'calcRmax' in calcData.keys():
        rmax = calcData['calcRmax']
    else:
        rmax = np.tan(calcData['calcAngmax']/180/60*np.pi)*haloData['haloDistance']/(1+haloData['haloZ'])**2
    b_av,ne_av = physical_averages(rmax,mode_exp,calcData,haloData,magData,gasData)
    if calcData['electronMode'] == "green-python":
        print("=========================================================")
        print("Calculating Electron Equilibriumn Distributions via Green's Function with Python")
        print("=========================================================")
        print('Magnetic Field Average Strength: {:.2e} micro Gauss'.format(b_av))
        print('Gas Average Density: {:.2e} cm^-3'.format(ne_av))
        E_set = takeSamples(calcData['log10ESampleMinFactor'],0,calcData['eSampleNum'],spacing="lin")
        Q_set = partData['dNdxInterp']['positrons'](mxEff,E_set).flatten()/np.log(1e1)/10**E_set/mxEff*(constants.m_e*constants.c**2).to("GeV").value
        E_set = 10**E_set*mxEff/(constants.m_e*constants.c**2).to("GeV").value
        r_sample = [takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],haloData['haloRvir']*2,calcData['rSampleNum']),takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],haloData['haloRvir']*2,calcData['rGreenSampleNum'])]
        rho_dm_sample = [haloData['haloDensityFunc'](r_sample[0])**mode_exp,haloData['haloDensityFunc'](r_sample[1])**mode_exp]
        calcData['results']['electronData'][mIndex] = electron.equilibrium_electrons(E_set,Q_set,r_sample,rho_dm_sample,mx,mode_exp,b_av,ne_av,haloData['haloZ'],lc,delta,diff,d0,ISRF)
    elif calcData['electronMode'] == "green-c":
        print("=========================================================")
        print("Calculating Electron Equilibriumn Distributions via Green's Function with C")
        print("=========================================================")
        print('Magnetic Field Average Strength: {:.2e} micro Gauss'.format(b_av))
        print('Gas Average Density: {:.2e} cm^-3'.format(ne_av))
        E_set = takeSamples(calcData['log10ESampleMinFactor'],0,calcData['eSampleNum'],spacing="lin")
        Q_set = partData['dNdxInterp']['positrons'](mxEff,E_set).flatten()/np.log(1e1)/10**E_set/mxEff*(constants.m_e*constants.c**2).to("GeV").value
        E_set = 10**E_set*mxEff/(constants.m_e*constants.c**2).to("GeV").value
        r_sample = [takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],haloData['haloRvir']*2,calcData['rSampleNum']),takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],haloData['haloRvir']*2,calcData['rGreenSampleNum'])]
        rho_dm_sample = [haloData['haloDensityFunc'](r_sample[0])**mode_exp,haloData['haloDensityFunc'](r_sample[1])**mode_exp]
        b_sample = magData['magFieldFunc'](r_sample[0])
        ne_sample = gasData['gasDensityFunc'](r_sample[0])
        py_file = "temp_electrons_py.out"
        c_file = "temp_electrons_c.in"
        wd = os.getcwd()
        calcData['results']['electronData'][mIndex] = electron.electrons_from_c(join(wd,py_file),join(wd,c_file),calcData['electronExecFile'],E_set,Q_set,r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,haloData['haloZ'],lc,delta,diff,d0,ISRF,num_threads=calcData['threadNumber'])
        #os.remove(join(wd,py_file))
        #os.remove(join(wd,c_file))
        if calcData['results']['electronData'][mIndex] is None:
            fatal_error("The electron executable {} is not compiled or location not specified correctly".format(calcData['electronExecFile']))
    elif calcData['electronMode'] == "crank-python":
        print("=========================================================")
        print("Calculating Electron Equilibriumn Distributions via Crank-Nicolson with Python")
        print("=========================================================")
        E_set = 10**takeSamples(calcData['log10ESampleMinFactor'],0,calcData['eSampleNum'],spacing="lin")*mxEff
        Q_set = partData['dNdxInterp']['positrons'](mxEff,np.log10(E_set/mxEff)).flatten()/np.log(1e1)/E_set
        r_sample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],haloData['haloRvir']*2,calcData['rSampleNum'])
        rho_sample = astrophysics.haloDensityBuilder(haloData)(r_sample)
        b_sample = magData['magFieldFunc'](r_sample)
        ne_sample = gasData['gasDensityFunc'](r_sample)
        r = sympy.symbols('r')
        dBdr_sample = lambdify(r,sympy.diff(magData['magFieldFunc'](r),r))(r_sample)
        cnSolver = cn_electron.cn_scheme(benchmark_flag=calcData['crankBenchMarkMode'],const_Delta_t=calcData['crankDeltaTConstant'])
        calcData['results']['electronData'][mIndex] = cnSolver.solveElectrons(mx,haloData['haloZ'],E_set,r_sample,rho_sample,Q_set,b_sample,dBdr_sample,ne_sample,haloData['haloScale'],1.0,lossOnly=diffData['lossOnly'],mode_exp=mode_exp,Delta_ti=calcData['crankDeltaTi'],max_part_t=calcData['crankMaxSteps'],Delta_t_reduction=calcData['crankDeltaTReduction'])
    print("Process Complete")
    return calcData

def calcRadioEm(mx,calcData,haloData,partData,magData,gasData,diffData):
    """
    Computes radio emissivities from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calcData : dictionary
        Calculation properties
    haloData : dictionary
        Halo properties
    partData : dictionary
        Particle physics
    magData : dictionary
        Magnetic field
    gasData : dictionary
        Das distribution
    diffData : dictionary
        Diffusion properties

    Returns
    ---------------------------
    calcData : dictionary
        Calculation information with radio emissivity in calcData['results']['radioEmData']
    """
    print("=========================================================")
    print("Calculating Radio Emissivity")
    print("=========================================================")
    if partData['emModel'] == "annihilation":
        mode_exp = 2.0
    else:
        mode_exp = 1.0
    mxEff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mxEff = mx
    mIndex = getIndex(calcData['mWIMP'],mx)
    rSample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],2*haloData['haloRvir'],calcData['rSampleNum'])
    fSample = calcData['fSampleValues']
    xSample = takeSamples(calcData['log10ESampleMinFactor'],0,calcData['eSampleNum'],spacing="lin")
    gSample = 10**xSample*mxEff/(constants.m_e*constants.c**2).to("GeV").value
    bSample = magData['magFieldFunc'](rSample)
    neSample = gasData['gasDensityFunc'](rSample)
    if calcData['results']['electronData'][mIndex] is None:
        calcData = calcElectrons(mx,calcData,haloData,partData,magData,gasData,diffData)
    electrons = calcData['results']['electronData'][mIndex]
    calcData['results']['radioEmData'][mIndex] = emissivity.radioEmGrid(electrons,fSample,rSample,gSample,bSample,neSample)
    print("Process Complete")
    return calcData

def calcPrimaryEm(mx,calcData,haloData,partData):
    """
    Computes primary gamma-ray or neutrino emissivity from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calcData : dictionary
        Calculation properties
    haloData : dictionary
        Halo properties
    partData : dictionary
        Particle physics

    Returns
    ---------------------------
    calcData : dictionary
        Calculation information with emissivity in calcData['results'][x], x = primaryEmData or neutrinoEmData
    """
    if calcData['freqMode'] in ["gamma","pgamma","all"]:
        print("=========================================================")
        print("Calculating Primary Gamma-ray Emissivity")
        print("=========================================================")
        specType = "gammas"
        emmType = 'primaryEmData'
    else:
        print("=========================================================")
        print("Calculating Neutrino Emissivity")
        print("=========================================================")
        specType = calcData['freqMode']
        emmType = 'neutrinoEmData'
    mIndex = getIndex(calcData['mWIMP'],mx)
    if partData['emModel'] == "annihilation":
        mode_exp = 2.0
    else:
        mode_exp = 1.0
    mxEff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mxEff = mx
    rSample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],haloData['haloRvir']*2,calcData['rSampleNum'])
    fSample = calcData['fSampleValues']
    xSample = takeSamples(calcData['log10ESampleMinFactor'],0,calcData['eSampleNum'],spacing="lin")
    gSample = 10**xSample*mxEff/(constants.m_e*constants.c**2).to("GeV").value
    rhoSample = haloData['haloDensityFunc'](rSample)
    qSample = partData['dNdxInterp'][specType](mxEff,xSample).flatten()/np.log(1e1)/10**xSample/mxEff*(constants.m_e*constants.c**2).to("GeV").value
    calcData['results'][emmType][mIndex] = emissivity.primaryEmHighE(mx,rhoSample,haloData['haloZ'],gSample,qSample,fSample,mode_exp)
    print("Process Complete")
    return calcData

def calcSecondaryEm(mx,calcData,haloData,partData,magData,gasData,diffData):
    """
    Computes secondary high-energy emissivities from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calcData : dictionary
        Calculation properties
    haloData : dictionary
        Halo properties
    partData : dictionary
        Particle physics
    magData : dictionary
        Magnetic field
    gasData : dictionary
        Das distribution
    diffData : dictionary
        Diffusion properties

    Returns
    ---------------------------
    calcData : dictionary
        Calculation information with emissivity in calcData['results']['secondaryEmData']
    """
    print("=========================================================")
    print("Calculating Secondary Gamma-ray Emissivity")
    print("=========================================================")
    if partData['emModel'] == "annihilation":
        mode_exp = 2.0
    else:
        mode_exp = 1.0
    mIndex = getIndex(calcData['mWIMP'],mx)
    mxEff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mxEff = mx
    rSample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],haloData['haloRvir']*2,calcData['rSampleNum'])
    fSample = calcData['fSampleValues'] #frequency values
    xSample = takeSamples(calcData['log10ESampleMinFactor'],0,calcData['eSampleNum'],spacing="lin") 
    gSample = 10**xSample*mxEff/(constants.m_e*constants.c**2).to("GeV").value
    neSample = gasData['gasDensityFunc'](rSample)
    if calcData['results']['electronData'][mIndex] is None:
        calcData = calcElectrons(mx,calcData,haloData,partData,magData,gasData,diffData)
    electrons = calcData['results']['electronData'][mIndex]
    calcData['results']['secondaryEmData'][mIndex] = emissivity.secondaryEmHighE(electrons,haloData['haloZ'],gSample,fSample,neSample)
    print("Process Complete")
    return calcData  

def calcFlux(mx,calcData,haloData):
    """
    Computes flux from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calcData : dictionary
        Calculation properties
    haloData : dictionary
        Halo properties

    Returns
    ---------------------------
    calcData : dictionary
        Calculation information with flux in calcData['results']['finalData']
    """
    print("=========================================================")
    print("Calculating Flux")
    print("=========================================================")
    print("Frequency mode: {}".format(calcData['freqMode']))
    if 'calcRmax' in calcData.keys():
        if calcData['calcRmax'] == "Rmax" or calcData['calcRmax'] == -1:
            rmax = haloData['haloRvir']
        else:
            rmax = calcData['calcRmax']
        print("Integration radius: {} Mpc".format(rmax))
        
    else:
        rmax = np.tan(calcData['calcAngmax']/180/60*np.pi)*haloData['haloDistance']/(1+haloData['haloZ'])**2
        print("Integration radius: {} arcmins = {} Mpc".format(calcData['calcAngmax'],rmax))
    mIndex = getIndex(calcData['mWIMP'],mx)
    if calcData['freqMode'] == "all":
        emm = calcData['results']['radioEmData'][mIndex] + calcData['results']['primaryEmData'][mIndex] + calcData['results']['secondaryEmData'][mIndex]
    elif calcData['freqMode'] == "gamma":
        emm = calcData['results']['primaryEmData'][mIndex] + calcData['results']['secondaryEmData'][mIndex]
    elif calcData['freqMode'] == "pgamma":
        emm = calcData['results']['primaryEmData'][mIndex]
    elif calcData['freqMode'] == "sgamma":
        emm = calcData['results']['secondaryEmData'][mIndex]
    elif calcData['freqMode'] == "radio":
        emm = calcData['results']['radioEmData'][mIndex] 
    elif "neutrinos" in calcData['freqMode']:
        emm = calcData['results']['neutrinoEmData'][mIndex]
    rSample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],haloData['haloRvir']*2,calcData['rSampleNum'])
    fSample = calcData['fSampleValues']
    calcData['results']['finalData'][mIndex] = fluxes.fluxGrid(rmax,haloData['haloDistance'],fSample,rSample,emm,boostMod=1.0)
    print("Process Complete")
    return calcData

def calcSB(mx,calcData,haloData):
    """
    Computes surface brightness from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calcData : dictionary
        Calculation properties
    haloData : dictionary
        Halo properties

    Returns
    ---------------------------
    calcData : dictionary
        Calculation information with surface brightness in calcData['results']['finalData']
    """
    print("=========================================================")
    print("Calculating Surface Brightness")
    print("=========================================================")
    print("Frequency mode: {}".format(calcData['freqMode']))
    mIndex = getIndex(calcData['mWIMP'],mx)
    if calcData['freqMode'] == "all":
        emm = calcData['results']['radioEmData'][mIndex] + calcData['results']['primaryEmData'][mIndex] + calcData['results']['secondaryEmData'][mIndex]
    elif calcData['freqMode'] == "gamma":
        emm = calcData['results']['primaryEmData'][mIndex] + calcData['results']['secondaryEmData'][mIndex]
    elif calcData['freqMode'] == "pgamma":
        emm = calcData['results']['primaryEmData'][mIndex]
    elif calcData['freqMode'] == "sgamma":
        emm = calcData['results']['secondaryEmData'][mIndex]
    elif calcData['freqMode'] == "radio":
        emm = calcData['results']['radioEmData'][mIndex] 
    elif "neutrinos" in calcData['freqMode']:
        emm = calcData['results']['neutrinoEmData'][mIndex]
    nuSB = []
    rSample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],haloData['haloRvir']*2,calcData['rSampleNum'])
    fSample = calcData['fSampleValues']
    for nu in fSample:
        nuSB.append(fluxes.surfaceBrightnessLoop(nu,fSample,rSample,emm)[1])
    calcData['results']['finalData'][mIndex] = np.array(nuSB)
    calcData['angSampleValues'] = np.arctan(takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],haloData['haloRvir']*2,calcData['rSampleNum'])/haloData['haloDistance']*(1+haloData['haloZ'])**2)/np.pi*180*60
    print("Process Complete")
    return calcData  

def calcJFlux(mx,calcData,haloData,partData):
    """
    Computes J/D-factor flux from given dictionaries

    Arguments
    ---------------------------
    mx : float
        WIMP mass in GeV
    calcData : dictionary
        Calculation properties
    haloData : dictionary
        Halo properties
    partData : dictionary
        Particle physics

    Returns
    ---------------------------
    calcData : dictionary
        Calculation information with flux in calcData['results']['finalData']
    """
    print("=========================================================")
    print("Calculating Flux From J/D-factor")
    print("=========================================================")
    print("Frequency mode: {}".format(calcData['freqMode']))
    if (not 'haloJFactor' in haloData.keys()) and partData['emModel'] == "annihilation":
        fatal_error("haloData parameter haloJFactor must be supplied to find a jflux for emModel = annihilation")  
    elif (not 'haloDFactor' in haloData.keys()) and partData['emModel'] == "decay":
        fatal_error("haloData parameter haloDFactor must be supplied to find a jflux for emModel = decay")  
    else:
        if partData['emModel'] == "annihilation":
            jFac = haloData['haloJFactor']
            mode_exp = 2.0
        else:
            jFac = haloData['haloDFactor']
            mode_exp = 1.0
        mxEff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mxEff = mx
        if calcData['freqMode'] == "pgamma":
            specType = "gammas"
        else:
            specType = calcData['freqMode']
        mIndex = getIndex(calcData['mWIMP'],mx)
        fSample = calcData['fSampleValues']
        xSample = takeSamples(calcData['log10ESampleMinFactor'],0,calcData['eSampleNum'],spacing="lin")
        gSample = 10**xSample*mxEff/(constants.m_e*constants.c**2).to("GeV").value  
        qSampleGamma = partData['dNdxInterp'][specType](mxEff,xSample).flatten()/np.log(1e1)/10**xSample/mxEff*(constants.m_e*constants.c**2).to("GeV").value
        calcData['results']['finalData'][mIndex] = fluxes.fluxFromJFactor(mx,haloData['haloZ'],jFac,fSample,gSample,qSampleGamma,mode_exp)
    print("Process Complete")
    return calcData

def runCalculation(calcData,haloData,partData,magData,gasData,diffData,cosmoData,overWrite=False):
    """
    Processes dictionaries and runs the requested operations

    Arguments
    ---------------------------
    calcData : dictionary
        Calculation properties
    haloData : dictionary
        Halo properties
    partData : dictionary
        Particle physics
    magData : dictionary
        Magnetic field
    gasData : dictionary
        Das distribution
    diffData : dictionary
        Diffusion properties
    diffData : dictionary
        Diffusion properties
    cosmoData : dictionary
        Cosmology properties

    Returns
    ---------------------------
    All given dictionaries checked and updated, including calcData with completed calcData['results']
    """
    cosmoData = checkCosmology(cosmoData)
    if not calcData['calcMode'] == "jflux":
        if not calcData['freqMode'] == "pgamma":
            magData = checkMagnetic(magData)
            gasData = checkGas(gasData)
            diffData = checkDiffusion(diffData)
        haloData = checkHalo(haloData,cosmoData)
    calcData = checkCalculation(calcData)
    partData = checkParticles(partData,calcData)
    print("=========================================================")
    print("Beginning DarkMatters calculations")
    print("=========================================================")
    print("Frequency mode: {}".format(calcData['freqMode']))
    print("Calculation type: {}".format(calcData['calcMode']))
    calcWrite(calcData,haloData,partData,magData,gasData,diffData)
    for mx in calcData['mWIMP']:
        wimpWrite(mx,partData)
        mIndex = getIndex(calcData['mWIMP'],mx)
        if not calcData['calcMode'] == "jflux":
            if (calcData['results']['electronData'][mIndex] is None or overWrite) and (not calcData['freqMode'] == "pgamma") and (not "neutrinos" in calcData['freqMode']):
                calcData = calcElectrons(mx,calcData,haloData,partData,magData,gasData,diffData)
            if calcData['freqMode'] in ["all","radio"] and (calcData['results']['radioEmData'][mIndex] is None or overWrite):
                calcData = calcRadioEm(mx,calcData,haloData,partData,magData,gasData,diffData)
            if calcData['freqMode'] in ["all","gamma","pgamma"] and (calcData['results']['primaryEmData'][mIndex] is None or overWrite):
                calcData = calcPrimaryEm(mx,calcData,haloData,partData)
            if calcData['freqMode'] in ["all","gamma","sgamma"] and (calcData['results']['secondaryEmData'][mIndex] is None or overWrite):
                calcData = calcSecondaryEm(mx,calcData,haloData,partData,magData,gasData,diffData)
            if "neutrinos" in calcData['freqMode']:
                calcData = calcPrimaryEm(mx,calcData,haloData,partData)
            if calcData['calcMode'] == "flux":
                calcData = calcFlux(mx,calcData,haloData)
            elif calcData['calcMode'] == "sb":
                calcData = calcSB(mx,calcData,haloData)
        else:
            calcData = calcJFlux(mx,calcData,haloData,partData)
    py_file = "temp_electrons_py.out"
    c_file = "temp_electrons_c.in"
    wd = os.getcwd()
    if isfile(join(wd,py_file)):
        os.remove(join(wd,py_file))
    if isfile(join(wd,c_file)):
        os.remove(join(wd,c_file))
    return calcData,haloData,partData,magData,gasData,diffData,cosmoData


