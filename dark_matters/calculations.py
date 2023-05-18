"""
DarkMatters module for controlling execution of calculations
"""
import numpy as np
import os
from scipy.integrate import simpson
from astropy import constants
import sympy
from sympy.utilities.lambdify import lambdify

from .output import fatal_error,warning,calcWrite,wimpWrite,spacer_length
from .dictionary_checks import checkCosmology,checkCalculation,checkDiffusion,checkGas,checkHalo,checkMagnetic,checkParticles
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

def takeSamples(xmin,xmax,nx,spacing="log",axis=-1):
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
        print("="*spacer_length)
        print(f"Electron Equilibrium distribution exists for WIMP mass {mx} GeV and overWrite=False, skipping")
        print("="*spacer_length)
        print("Process Complete")
        return calcData
    if partData['emModel'] == "annihilation":
        mode_exp = 2.0
    else:
        mode_exp = 1.0
    if partData['decayInput']:
        mxEff = mx
    else:
        mxEff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mxEff = mx
    if diffData['lossOnly']:
        diff = 0
        delta = 0
        d0 = diffData['diffConstant']
    else:
        diff = 1
        delta = diffData['diffIndex']
        d0 = diffData['diffConstant']
    if diffData['diffRmax'] == "2*Rvir":
        rLimit = 2*haloData['haloRvir']
    else:
        rLimit = diffData['diffRmax']

    if "green" in calcData['electronMode']:
        b_av,ne_av = physical_averages(haloData['greenAveragingScale'],mode_exp,calcData,haloData,magData,gasData)
        if "gasAverageDensity" in gasData.keys():
            ne_av = gasData['gasAverageDensity']
        if "magFieldAverage" in magData.keys():
            b_av = gasData['magFieldAverage']
    elif "greenAveragingScale" in haloData.keys():
        haloData.pop('greenAveragingScale')
        
    if partData['emModel'] == "annihilation":
        sigV = partData['crossSection']
    else:
        sigV = partData['decayRate']

    r_sample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum'])
    rho_dm_sample = haloData['haloDensityFunc'](r_sample)
    b_sample = magData['magFieldFunc'](r_sample)
    ne_sample = gasData['gasDensityFunc'](r_sample)

    if "green" in calcData['electronMode']:
        #Note sigV is left out here to simplify numerical convergence, it is restored below
        E_set = takeSamples(np.log10(calcData['eSampleMin']/mxEff),0,calcData['eSampleNum'],spacing="lin")
        Q_set = partData['dNdxInterp']['positrons'](mxEff,E_set).flatten()/np.log(1e1)/10**E_set/mxEff*(constants.m_e*constants.c**2).to("GeV").value
        E_set = 10**E_set*mxEff/(constants.m_e*constants.c**2).to("GeV").value
    else:
        E_set = 10**takeSamples(np.log10(calcData['eSampleMin']/mxEff),0,calcData['eSampleNum'],spacing="lin")*mxEff
        Q_set = partData['dNdxInterp']['positrons'](mxEff,np.log10(E_set/mxEff)).flatten()/np.log(1e1)/E_set*sigV

    if np.all(Q_set == 0.0):
        warning("At WIMP mass {mx} GeV dN/dE functions are zero at all considered energies!\nNote that in decay cases we sample mxEff= 0.5*mx")

    print("="*spacer_length)
    print("Calculating Electron Equilibriumn Distributions")
    print("="*spacer_length)
    if calcData['electronMode'] == "green-python":
        print("Solution via: Green's function (python implementation)")
        print(f'Magnetic Field Average Strength: {b_av:.2e} micro Gauss')
        print(f'Gas Average Density: {ne_av:.2e} cm^-3')
        print(f'Averaging scale: {haloData["greenAveragingScale"]:.2e} Mpc')
        calcData['results']['electronData'][mIndex] = green_electron.equilibriumElectronsGridPartial(calcData['eGreenSampleNum'],E_set,Q_set,calcData['rGreenSampleNum'],r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,haloData['haloZ'],delta,diff,d0,diffData["photonDensity"],calcData['threadNumber'],calcData['imageNumber'])*sigV
    elif calcData['electronMode'] == "green-c":
        print("Solution via: Green's function (c++ implementation)")
        print(f'Magnetic Field Average Strength: {b_av:.2e} micro Gauss')
        print(f'Gas Average Density: {ne_av:.2e} cm^-3')
        print(f'Averaging scale: {haloData["greenAveragingScale"]:.2e} Mpc')
        py_file = "temp_electrons_py.out"
        c_file = "temp_electrons_c.in"
        wd = os.getcwd()
        calcData['results']['electronData'][mIndex] = green_electron.electrons_from_c(os.path.join(wd,py_file),os.path.join(wd,c_file),calcData['electronExecFile'],calcData['eGreenSampleNum'],E_set,Q_set,calcData['rGreenSampleNum'],r_sample,rho_dm_sample,b_sample,ne_sample,mx,mode_exp,b_av,ne_av,haloData['haloZ'],delta,diff,d0,diffData['photonDensity'],num_threads=calcData['threadNumber'],num_images=calcData['imageNumber'])
        if calcData['results']['electronData'][mIndex] is None:
            fatal_error(f"The electron executable {calcData['electronExecFile']} is not compiled or location not specified correctly")
        else:
            calcData['results']['electronData'][mIndex] *= sigV
    elif calcData['electronMode'] == "adi-python":
        print("Solution via: ADI method (python implementation)")
        r = sympy.symbols('r')
        dBdr_sample = lambdify(r,sympy.diff(magData['magFieldFunc'](r),r))(r_sample)
        if np.isscalar(dBdr_sample):
            dBdr_sample = dBdr_sample*np.ones_like(r_sample)
        adiSolver = adi_electron.adi_scheme(benchmark_flag=calcData['adiBenchMarkMode'],const_Delta_t=calcData['adiDeltaTConstant'])
        calcData['results']['electronData'][mIndex] = adiSolver.solveElectrons(mx,haloData['haloZ'],E_set,r_sample,rho_dm_sample,Q_set,b_sample,dBdr_sample,ne_sample,haloData['haloScale'],1.0,diffData['diffIndex'],uPh=diffData['photonDensity'],diff0=diffData['diffConstant'],Delta_t_min=calcData['adiDeltaTMin'],lossOnly=diffData['lossOnly'],mode_exp=mode_exp,Delta_ti=calcData['adiDeltaTi'],max_t_part=calcData['adiMaxSteps'],Delta_t_reduction=calcData['adiDeltaTReduction'])*(constants.m_e*constants.c**2).to("GeV").value
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
    print("="*spacer_length)
    print("Calculating Radio Emissivity")
    print("="*spacer_length)
    if partData['emModel'] == "annihilation":
        mode_exp = 2.0
    else:
        mode_exp = 1.0
    if diffData['diffRmax'] == "2*Rvir":
        rLimit = 2*haloData['haloRvir']
    else:
        rLimit = diffData['diffRmax']
    if partData['decayInput']:
        mxEff = mx
    else:
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
    if calcData['results']['radioEmData'][mIndex] is None:
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
        print("="*spacer_length)
        print("Calculating Primary Gamma-ray Emissivity")
        print("="*spacer_length)
        specType = "gammas"
        emmType = 'primaryEmData'
    else:
        print("="*spacer_length)
        print("Calculating Neutrino Emissivity")
        print("="*spacer_length)
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
    if partData['decayInput']:
        mxEff = mx
    else:
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
    if np.all(qSample == 0.0):
        warning("At WIMP mass {mx} GeV dN/dE functions are zero at all considered energies!\nNote that in decay cases we sample mxEff= 0.5*mx")
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
    print("="*spacer_length)
    print("Calculating Secondary Gamma-ray Emissivity")
    print("="*spacer_length)
    if partData['emModel'] == "annihilation":
        mode_exp = 2.0
    else:
        mode_exp = 1.0
    mIndex = getIndex(calcData['mWIMP'],mx)
    if diffData['diffRmax'] == "2*Rvir":
        rLimit = 2*haloData['haloRvir']
    else:
        rLimit = diffData['diffRmax']
    if partData['decayInput']:
        mxEff = mx
    else:
        mxEff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mxEff = mx
    rSample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum'])
    fSample = calcData['fSampleValues'] #frequency values
    xSample = takeSamples(np.log10(calcData['eSampleMin']/mxEff),0,calcData['eSampleNum'],spacing="lin")
    gSample = 10**xSample*mxEff/(constants.m_e*constants.c**2).to("GeV").value
    neSample = gasData['gasDensityFunc'](rSample)
    if calcData['results']['electronData'][mIndex] is None:
        calcData = calcElectrons(mx,calcData,haloData,partData,magData,gasData,diffData)
    electrons = calcData['results']['electronData'][mIndex]
    calcData['results']['secondaryEmData'][mIndex] = emissivity.secondaryEmHighE(electrons,haloData['haloZ'],gSample,fSample,neSample,diffData['photonTemp'])
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
    print("="*spacer_length)
    print("Calculating Flux")
    print("="*spacer_length)
    print(f"Frequency mode: {calcData['freqMode']}")
    if diffData['diffRmax'] == "2*Rvir":
        rLimit = 2*haloData['haloRvir']
    else:
        rLimit = diffData['diffRmax']
    if 'calcRmaxIntegrate' in calcData.keys():
        if calcData['calcRmaxIntegrate'] == "Rvir":
            rmax = haloData['haloRvir']
        elif calcData['calcRmaxIntegrate'] == -1:
            rmax = rLimit
        else:
            rmax = calcData['calcRmaxIntegrate']
        print(f"Integration radius: {rmax} Mpc")
    else:
        rmax = np.tan(calcData['calcAngmaxIntegrate']/180/60*np.pi)*haloData['haloDistance']/(1+haloData['haloZ'])**2
        print(f"Integration radius: {calcData['calcAngmaxIntegrate']} arcmins = {rmax} Mpc")
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
    rSample = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum'])
    fSample = calcData['fSampleValues']
    calcData['results']['finalData'][mIndex] = fluxes.fluxGrid(rmax,haloData['haloDistance'],fSample,rSample,emm,boostMod=1.0,ergs=calcData["outCGS"])
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
    print("="*spacer_length)
    print("Calculating Surface Brightness")
    print("="*spacer_length)
    print(f"Frequency mode: {calcData['freqMode']}")
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
        nuSB.append(fluxes.surfaceBrightnessLoop(nu,fSample,rSample,emm,ergs=calcData["outCGS"])[1])
    calcData['results']['finalData'][mIndex] = np.array(nuSB)
    calcData['results']['angSampleValues'] = np.arctan(takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum'])/haloData['haloDistance']*(1+haloData['haloZ'])**2)/np.pi*180*60
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
    print("="*spacer_length)
    print("Calculating Flux From J/D-factor")
    print("="*spacer_length)
    print(f"Frequency mode: {calcData['freqMode']}")
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
        if partData['decayInput']:
            mxEff = mx
        else:
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
    """
    Processes dictionaries to prepare for calculations, will crash if this is not possible

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
    clear : string (optional)
        What results to clear, can be 'all', 'observables' or 'final' (defaults to 'all')

    Returns
    ---------------------------
    All given dictionaries checked and ready for calculations
    """
    cosmoData = checkCosmology(cosmoData)
    if not calcData['calcMode'] == "jflux":
        if (not calcData['freqMode'] == "pgamma") and (not "neutrinos" in calcData['freqMode']):
            magData = checkMagnetic(magData)
            gasData = checkGas(gasData)
        diffData = checkDiffusion(diffData)
        haloData = checkHalo(haloData,cosmoData)
    else:
        haloData = checkHalo(haloData,cosmoData,minimal=True)
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
                    fatal_error("calculations.runChecks(): You cannot run with clear=observables if your calcData dictionary has incomplete electronData")
            else:
                fatal_error("calculations.runChecks(): You cannot run with clear=observables if your calcData dictionary has no existing electronData")
        else:
            fatal_error("calculations.runChecks(): You cannot run with clear=observables if your calcData dictionary has no existing results")
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
                    fatal_error("calculations.runChecks(): You cannot run with clear=final if your calcData dictionary has incomplete electronData")
            else:
                fatal_error("calculations.runChecks(): You cannot run with clear=final if your calcData dictionary has no existing electronData")
        else:
            fatal_error("calculations.runChecks(): You cannot run with clear=observables if your calcData dictionary has no existing results")
        calcData['results']['finalData'] = []
        for i in range(len(calcData['mWIMP'])):
            calcData['results']['finalData'].append(None)
    resultUnits = {"electronData":"GeV/cm^3","radioEmData":"GeV/cm^3","primaryEmData":"GeV/cm^3","secondaryEmData":"GeV/cm^3","fSampleValues":"MHz"}
    if "flux" in calcData['calcMode']:
        if calcData['outCGS']:
            resultUnits['finalData'] = "erg/(cm^2 s)"
        else:
            resultUnits['finalData'] = "Jy"
    else:
        if calcData['outCGS']:
            resultUnits['finalData'] = "erg/(cm^2 s arcmin^2)"
        else:
            resultUnits['finalData'] = "Jy/arcmin^2"
        resultUnits['angSampleValues'] = "arcmin"
    calcData['results']['units'] = resultUnits
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
    overWriteElectrons : boolean (optional)
        if False will not overWrite existing electronData values (defaults to True)
    clear : string (optional)
        What results to clear, can be 'all', 'observables' or 'final' (defaults to 'all')

    Returns
    ---------------------------
    All given dictionaries checked and updated, including calcData with completed calcData['results']
    """
    
    calcData,haloData,partData,magData,gasData,diffData,cosmoData = runChecks(calcData,haloData,partData,magData,gasData,diffData,cosmoData,clear)
    print("="*spacer_length)
    print("Beginning DarkMatters calculations")
    print("="*spacer_length)
    print(f"Frequency mode: {calcData['freqMode']}")
    print(f"Calculation type: {calcData['calcMode']}")
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
            if diffData['diffRmax'] == "2*Rvir":
                rLimit = 2*haloData['haloRvir']
            else:
                rLimit = diffData['diffRmax']
            calcData['results']['rSampleValues'] = takeSamples(haloData['haloScale']*10**calcData['log10RSampleMinFactor'],rLimit,calcData['rSampleNum'])
            calcData['results']['units']['rSampleValues'] = "Mpc"
            if partData['emModel'] == "annihilation":
                mode_exp = 2.0
            else:
                mode_exp = 1.0
            if calcData['freqMode'] in ["all","gamma","sgamma","radio"]:
                if partData['decayInput']:
                    mxEff = mx
                else:
                    mxEff = mx*0.5*mode_exp #this takes into account decay when mode_exp = 1, annihilation mode_exp = 2 means mxEff = mx
                xSample = takeSamples(np.log10(calcData['eSampleMin']/mxEff),0,calcData['eSampleNum'],spacing="lin")
                calcData['results']['eSampleValues'] = 10**xSample*mxEff
                calcData['results']['units']['eSampleValues'] = "GeV"
        else:
            calcData = calcJFlux(mx,calcData,haloData,partData)
    calcData['results']['fSampleValues'] = calcData['fSampleValues']
    py_file = "temp_electrons_py.out"
    c_file = "temp_electrons_c.in"
    wd = os.getcwd()
    if os.path.isfile(os.path.join(wd,py_file)):
        os.remove(os.path.join(wd,py_file))
    if os.path.isfile(os.path.join(wd,c_file)):
        os.remove(os.path.join(wd,c_file))
    return {'calcData':calcData,'haloData':haloData,'partData':partData,'magData':magData,'gasData':gasData,'diffData':diffData,'cosmoData':cosmoData}


