from genericpath import isfile
import numpy as np
from os.path import join
import os
from scipy.integrate import simpson
from astropy import constants
import sympy
from sympy.utilities.lambdify import lambdify

from .output import fatal_error,calcWrite,wimpWrite
from .dictionary_checks import checkCosmology,checkCalculation,checkDiffusion,checkGas,checkHalo,checkMagnetic,checkParticles
from .astro_cosmo import astrophysics
from .emissions import adi_electron,green_electron,fluxes,emissivity

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
        return simpson(y*w*r**2,r)/simpson(w*r**2,r)
    rSet = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rmax,100)
    if haloData['haloWeights'] == "rho":
        weights = haloData['haloDensityFunc'](rSet)**mode_exp #the average is weighted
    else:
        weights = np.ones_like(rSet)
    return weightedVolAvg(magData['magFieldFunc'](rSet),rSet,w=weights),weightedVolAvg(gasData['gasDensityFunc'](rSet),rSet,w=weights)


def calcElectrons(mx,calcData,haloData,partData,magData,gasData,diffData,overWrite=True):
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
    overWrite : boolean
        If True will replace any existing values in calcData['results']

    Returns
    ---------------------------
    calcData : dictionary
        Calculation information with electron distribution in calcData['results']['electronData']
    """
    mIndex = getIndex(calcData['mWIMP'],mx)
    if (not calcData['results']['electronData'][mIndex] is None) and (not overWrite):
        print("=========================================================")
        print(f"Electron Equilibrium distribution exists for WIMP mass {mx} GeV and overWrite=False, skipping")
        print("=========================================================")
        print("Process Complete")
        return calcData
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
        d0 = diffData['diffConstant']
    else:
        diff = 1
        lc = diffData['coherenceScale']*1e3 #kpc
        delta = diffData['diffIndex']
        d0 = diffData['diffConstant']
    if diffData['diffRmax'] == "2*Rvir":
        rLimit = 2*haloData['haloRvir']
    else:
        rLimit = diffData['diffRmax']
    b_av,ne_av = physical_averages(haloData['greenAveragingScale'],mode_exp,calcData,haloData,magData,gasData)
    if "gasAverageDensity" in gasData.keys():
        ne_av = gasData['gasAverageDensity']
    if "magFieldAverage" in magData.keys():
        b_av = gasData['magFieldAverage']
    if partData['emModel'] == "annihilation":
        sigV = partData['crossSection']
    else:
        sigV = partData['decayRate']
    if calcData['electronMode'] == "green-python":
        print("=========================================================")
        print("Calculating Electron Equilibriumn Distributions via Green's Function with Python")
        print("=========================================================")
        print('Magnetic Field Average Strength: {:.2e} micro Gauss'.format(b_av))
        print('Gas Average Density: {:.2e} cm^-3'.format(ne_av))
        E_set = takeSamples(np.log10(calcData['eSampleMin']/mxEff),0,calcData['eSampleNum'],spacing="lin")
        Q_set = partData['dNdxInterp']['positrons'](mxEff,E_set).flatten()/np.log(1e1)/10**E_set/mxEff*(constants.m_e*constants.c**2).to("GeV").value
        E_set = 10**E_set*mxEff/(constants.m_e*constants.c**2).to("GeV").value
        r_sample = [takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum']),takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rGreenSampleNum'])]
        rho_dm_sample = [haloData['haloDensityFunc'](r_sample[0]),haloData['haloDensityFunc'](r_sample[1])]
        b_sample = magData['magFieldFunc'](r_sample[0])
        ne_sample = gasData['gasDensityFunc'](r_sample[0])
        calcData['results']['electronData'][mIndex] = green_electron.equilibriumElectronsGridPartial(E_set,Q_set,r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,haloData['haloZ'],lc,delta,diff,d0,ISRF,calcData['threadNumber'],calcData['imageNumber'])*sigV
    elif calcData['electronMode'] == "green-c":
        print("=========================================================")
        print("Calculating Electron Equilibriumn Distributions via Green's Function with C")
        print("=========================================================")
        print('Magnetic Field Average Strength: {:.2e} micro Gauss'.format(b_av))
        print('Gas Average Density: {:.2e} cm^-3'.format(ne_av))
        E_set = takeSamples(np.log10(calcData['eSampleMin']/mxEff),0,calcData['eSampleNum'],spacing="lin")
        Q_set = partData['dNdxInterp']['positrons'](mxEff,E_set).flatten()/np.log(1e1)/10**E_set/mxEff*(constants.m_e*constants.c**2).to("GeV").value
        E_set = 10**E_set*mxEff/(constants.m_e*constants.c**2).to("GeV").value
        r_sample = [takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum']),takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rGreenSampleNum'])]
        rho_dm_sample = [haloData['haloDensityFunc'](r_sample[0])**mode_exp,haloData['haloDensityFunc'](r_sample[1])**mode_exp]
        b_sample = magData['magFieldFunc'](r_sample[0])
        ne_sample = gasData['gasDensityFunc'](r_sample[0])
        py_file = "temp_electrons_py.out"
        c_file = "temp_electrons_c.in"
        wd = os.getcwd()
        calcData['results']['electronData'][mIndex] = green_electron.electrons_from_c(join(wd,py_file),join(wd,c_file),calcData['electronExecFile'],calcData['eGreenSampleNum'],E_set,Q_set,r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,haloData['haloZ'],lc,delta,diff,d0,ISRF,num_threads=calcData['threadNumber'],num_images=calcData['imageNumber'])*sigV
        if calcData['results']['electronData'][mIndex] is None:
            fatal_error("The electron executable {} is not compiled or location not specified correctly".format(calcData['electronExecFile']))
    elif calcData['electronMode'] == "adi-python":
        print("=========================================================")
        print("Calculating Electron Equilibriumn Distributions via ADI method with Python")
        print("=========================================================")
        E_set = 10**takeSamples(np.log10(calcData['eSampleMin']/mxEff),0,calcData['eSampleNum'],spacing="lin")*mxEff
        Q_set = partData['dNdxInterp']['positrons'](mxEff,np.log10(E_set/mxEff)).flatten()/np.log(1e1)/E_set*sigV
        r_sample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum'])
        rho_sample = astrophysics.haloDensityBuilder(haloData)(r_sample)
        b_sample = magData['magFieldFunc'](r_sample)
        ne_sample = gasData['gasDensityFunc'](r_sample)
        r = sympy.symbols('r')
        dBdr_sample = lambdify(r,sympy.diff(magData['magFieldFunc'](r),r))(r_sample)
        if np.isscalar(dBdr_sample):
            dBdr_sample = dBdr_sample*np.ones_like(r_sample)
        adiSolver = adi_electron.adi_scheme(benchmark_flag=calcData['adiBenchMarkMode'],const_Delta_t=calcData['adiDeltaTConstant'])
        calcData['results']['electronData'][mIndex] = adiSolver.solveElectrons(mx,haloData['haloZ'],haloData['haloRvir'],E_set,r_sample,rho_sample,Q_set,b_sample,dBdr_sample,ne_sample,haloData['haloScale'],1.0,lc,diffData['diffIndex'],diff0=diffData['diffConstant'],Delta_t_min=calcData['adiDeltaTMin'],lossOnly=diffData['lossOnly'],mode_exp=mode_exp,Delta_ti=calcData['adiDeltaTi'],max_t_part=calcData['adiMaxSteps'],Delta_t_reduction=calcData['adiDeltaTReduction'])*(constants.m_e*constants.c**2).to("GeV").value
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
    if diffData['diffRmax'] == "2*Rvir":
        rLimit = 2*haloData['haloRvir']
    else:
        rLimit = diffData['diffRmax']
    mxEff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mxEff = mx
    mIndex = getIndex(calcData['mWIMP'],mx)
    rSample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum'])
    fSample = calcData['fSampleValues']
    xSample = takeSamples(np.log10(calcData['eSampleMin']/mxEff),0,calcData['eSampleNum'],spacing="lin")
    gSample = 10**xSample*mxEff/(constants.m_e*constants.c**2).to("GeV").value
    bSample = magData['magFieldFunc'](rSample)
    neSample = gasData['gasDensityFunc'](rSample)
    if calcData['results']['electronData'][mIndex] is None:
        calcData = calcElectrons(mx,calcData,haloData,partData,magData,gasData,diffData)
    electrons = calcData['results']['electronData'][mIndex]
    calcData['results']['radioEmData'][mIndex] = emissivity.radioEmGrid(electrons,fSample,rSample,gSample,bSample,neSample)
    print("Process Complete")
    return calcData

def calcPrimaryEm(mx,calcData,haloData,partData,diffData):
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
    if diffData['diffRmax'] == "2*Rvir":
        rLimit = 2*haloData['haloRvir']
    else:
        rLimit = diffData['diffRmax']
    mxEff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mxEff = mx
    rSample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum'])
    fSample = calcData['fSampleValues']
    xSample = takeSamples(np.log10(calcData['eSampleMin']/mxEff),0,calcData['eSampleNum'],spacing="lin")
    if partData['emModel'] == "annihilation":
        sigV = partData['crossSection']
    else:
        sigV = partData['decayRate']
    gSample = 10**xSample*mxEff/(constants.m_e*constants.c**2).to("GeV").value
    rhoSample = haloData['haloDensityFunc'](rSample)
    qSample = partData['dNdxInterp'][specType](mxEff,xSample).flatten()/np.log(1e1)/10**xSample/mxEff*(constants.m_e*constants.c**2).to("GeV").value*sigV
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
    if diffData['diffRmax'] == "2*Rvir":
        rLimit = 2*haloData['haloRvir']
    else:
        rLimit = diffData['diffRmax']
    mxEff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mxEff = mx
    rSample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum'])
    fSample = calcData['fSampleValues'] #frequency values
    xSample = takeSamples(np.log10(calcData['eSampleMin']/mxEff),0,calcData['eSampleNum'],spacing="lin")
    gSample = 10**xSample*mxEff/(constants.m_e*constants.c**2).to("GeV").value
    neSample = gasData['gasDensityFunc'](rSample)
    if calcData['results']['electronData'][mIndex] is None:
        calcData = calcElectrons(mx,calcData,haloData,partData,magData,gasData,diffData)
    electrons = calcData['results']['electronData'][mIndex]
    calcData['results']['secondaryEmData'][mIndex] = emissivity.secondaryEmHighE(electrons,haloData['haloZ'],gSample,fSample,neSample)
    print("Process Complete")
    return calcData  

def calcFlux(mx,calcData,haloData,diffData):
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
    if 'calcRmaxIntegrate' in calcData.keys():
        if calcData['calcRmaxIntegrate'] == "Rmax" or calcData['calcRmaxIntegrate'] == -1:
            rmax = haloData['haloRvir']
        else:
            rmax = calcData['calcRmaxIntegrate']
        print("Integration radius: {} Mpc".format(rmax))
        
    else:
        rmax = np.tan(calcData['calcAngmaxIntegrate']/180/60*np.pi)*haloData['haloDistance']/(1+haloData['haloZ'])**2
        print("Integration radius: {} arcmins = {} Mpc".format(calcData['calcAngmaxIntegrate'],rmax))
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
    if diffData['diffRmax'] == "2*Rvir":
        rLimit = 2*haloData['haloRvir']
    else:
        rLimit = diffData['diffRmax']
    rSample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum'])
    fSample = calcData['fSampleValues']
    calcData['results']['finalData'][mIndex] = fluxes.fluxGrid(rmax,haloData['haloDistance'],fSample,rSample,emm,boostMod=1.0)
    print("Process Complete")
    return calcData

def calcSB(mx,calcData,haloData,diffData):
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
    if diffData['diffRmax'] == "2*Rvir":
        rLimit = 2*haloData['haloRvir']
    else:
        rLimit = diffData['diffRmax']
    rSample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum'])
    fSample = calcData['fSampleValues']
    for nu in fSample:
        nuSB.append(fluxes.surfaceBrightnessLoop(nu,fSample,rSample,emm)[1])
    calcData['results']['finalData'][mIndex] = np.array(nuSB)
    calcData['angSampleValues'] = np.arctan(takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum'])/haloData['haloDistance']*(1+haloData['haloZ'])**2)/np.pi*180*60
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
        xSample = takeSamples(np.log10(calcData['eSampleMin']/mxEff),0,calcData['eSampleNum'],spacing="lin")
        gSample = 10**xSample*mxEff/(constants.m_e*constants.c**2).to("GeV").value  
        qSampleGamma = partData['dNdxInterp'][specType](mxEff,xSample).flatten()/np.log(1e1)/10**xSample/mxEff*(constants.m_e*constants.c**2).to("GeV").value*partData['crossSection']
        calcData['results']['finalData'][mIndex] = fluxes.fluxFromJFactor(mx,haloData['haloZ'],jFac,fSample,gSample,qSampleGamma,mode_exp)
    print("Process Complete")
    return calcData

def runChecks(calcData,haloData,partData,magData,gasData,diffData,cosmoData,clear):
    cosmoData = checkCosmology(cosmoData)
    if not calcData['calcMode'] == "jflux":
        if not calcData['freqMode'] == "pgamma":
            magData = checkMagnetic(magData)
            gasData = checkGas(gasData)
            diffData = checkDiffusion(diffData)
        haloData = checkHalo(haloData,cosmoData)
    calcData = checkCalculation(calcData)
    if clear == "all":
        calcData['results'] = {'electronData':[],'radioEmData':[],'primaryEmData':[],'secondaryEmData':[],'finalData':[],'neutrinoEmData':[]}
        for i in range(len(calcData['mWIMP'])):
            calcData['results']['electronData'].append(None)
            calcData['results']['radioEmData'].append(None)
            calcData['results']['primaryEmData'].append(None)
            calcData['results']['secondaryEmData'].append(None)
            calcData['results']['neutrinoEmData'].append(None)
            calcData['results']['finalData'].append(None)
    elif clear == "observables":
        if 'results' in calcData.keys():
            if 'electronData' in calcData['results'].keys():
                if np.any(calcData['results']['electronData'] is None):
                    fatal_error("You cannot run with clear=observables if your calcData dictionary has incomplete electronData")
            else:
                fatal_error("You cannot run with clear=observables if your calcData dictionary has no existing electronData")
        else:
            fatal_error("You cannot run with clear=observables if your calcData dictionary has no existing results")
        calcData['results']['radioEmData'] = []
        calcData['results']['primaryEmData'] = []
        calcData['results']['finalData'] = []
        calcData['results']['secondaryEmData'] = []
        calcData['results']['neutrinoEmData'] = []
        for i in range(len(calcData['mWIMP'])):
            calcData['results']['radioEmData'].append(None)
            calcData['results']['primaryEmData'].append(None)
            calcData['results']['secondaryEmData'].append(None)
            calcData['results']['neutrinoEmData'].append(None)
            calcData['results']['finalData'].append(None)
    else:
        if 'results' in calcData.keys():
            if 'electronData' in calcData['results'].keys():
                if np.any(calcData['results']['electronData'] is None):
                    fatal_error("You cannot run with clear=final if your calcData dictionary has incomplete electronData")
            else:
                fatal_error("You cannot run with clear=final if your calcData dictionary has no existing electronData")
        else:
            fatal_error("You cannot run with clear=observables if your calcData dictionary has no existing results")
        calcData['results']['finalData'] = []
        for i in range(len(calcData['mWIMP'])):
            calcData['results']['finalData'].append(None)
    partData = checkParticles(partData,calcData)
    return calcData,haloData,partData,magData,gasData,diffData,cosmoData

def runCalculation(calcData,haloData,partData,magData,gasData,diffData,cosmoData,overWriteElectrons=True,clear="all"):
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
    overWriteElectrons : boolean
        if False will not overWrite existing electronData values
    clear : string
        What results to clear, can be 'all', 'observables' or 'final'

    Returns
    ---------------------------
    All given dictionaries checked and updated, including calcData with completed calcData['results']
    """
    
    calcData,haloData,partData,magData,gasData,diffData,cosmoData = runChecks(calcData,haloData,partData,magData,gasData,diffData,cosmoData,clear)
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
            if (not calcData['freqMode'] == "pgamma") and (not "neutrinos" in calcData['freqMode']):
                calcData = calcElectrons(mx,calcData,haloData,partData,magData,gasData,diffData,overWrite=overWriteElectrons)
            if calcData['freqMode'] in ["all","radio"]:
                calcData = calcRadioEm(mx,calcData,haloData,partData,magData,gasData,diffData)
            if calcData['freqMode'] in ["all","gamma","pgamma"]:
                calcData = calcPrimaryEm(mx,calcData,haloData,partData,diffData)
            if calcData['freqMode'] in ["all","gamma","sgamma"]:
                calcData = calcSecondaryEm(mx,calcData,haloData,partData,magData,gasData,diffData)
            if "neutrinos" in calcData['freqMode']:
                calcData = calcPrimaryEm(mx,calcData,haloData,partData,diffData)
            if calcData['calcMode'] == "flux":
                calcData = calcFlux(mx,calcData,haloData,diffData)
            elif calcData['calcMode'] == "sb":
                calcData = calcSB(mx,calcData,haloData,diffData)
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


