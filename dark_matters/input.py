import numpy as np
import yaml
from astropy import units
from scipy.interpolate import interp2d
import os
from .output import fatal_error,checkQuant

def getSpectralData(spec_dir,partModel,specSet,mode="annihilation"):
    """
    Retrieves particle yield spectra for a given model, set of WIMP masses, and products

    Arguments
    ---------------------------
    spec_dir : str
        Path of folder where spectra are stored
    partModel : str 
        Label of particle physics model
    specSet :  str, list
        Particle yield spectra to be loadedAllowed entries are "gammas", "positrons", "neutrinos_x" where x = mu,e, or tau
    mode : str, optional 
        Annihilation or decay
    pppcdb4dm : bool
        Flag for using PPPCB4DM tables

    Returns
    ---------------------------
    specDict : dictionary
        Dictionary of yield spectra, keys matching specSet, values are interpolating functions
    """
    specDict = {}
    for f in specSet:
        if partModel in ["bb","qq","ww","ee","hh","tautau","mumu","tt","zz"]:
            specDict[f] = readSpectrum(os.path.join(spec_dir,"AtProduction_{}.dat".format(f)),partModel,mode=mode,pppcb4dm=True)
        else:
            specDict[f] = readSpectrum(os.path.join(spec_dir,"AtProduction_{}.dat".format(f)),partModel,mode=mode,pppcb4dm=False)
    return specDict

def readSpectrum(spec_file,partModel,mode="annihilation",pppcb4dm=True):
    """
    Reads file to get particle yield spectra for a given model and set of WIMP masses

    Arguments
    ---------------------------
    spec_file : str
        Path of spectrum file
    partModel : str 
        Label of particle physics model
    mode : float, optional 
        Flag, 2.0 for annihilation or 1.0 for decay
    pppcdb4dm : bool, optional 
        Flag for using PPPCDB4DM tables

    Returns
    ---------------------------
    intp: interpolating function (mx,log10(energy/mx))
        Interpolating function for particle yields

    Notes
    ---------------------------
    file names format : "AtProduction_partModel_products.dat", "products" can be "positrons", "gammas", or "neutrinos_e" etc 
    A custom spec_file must be formatted as follows:
    column 0: WIMP mass in GeV, column 1: log10(energy/mx) , column 2: dN/dlog10(energy/mx)
    """
    #mDM      Log[10,x]   eL         eR         e          \[Mu]L     \[Mu]R     \[Mu]      \[Tau]L    \[Tau]R    \[Tau]     q            c            b            t            WL          WT          W           ZL          ZT          Z           g            \[Gamma]    h           \[Nu]e     \[Nu]\[Mu]   \[Nu]\[Tau]   V->e       V->\[Mu]   V->\[Tau]
    chCols = {"ee":4,"mumu":7,"tautau":10,"qq":11,"bb":13,"tt":14,"ww":17,"zz":20,"gamma":22,'hh':23}
    if pppcb4dm:
        nCol = chCols[partModel]
    else:
        nCol = 2
    mCol = 0
    xCol = 1
    try:
        specData = np.loadtxt(spec_file,unpack=True)
    except IOError:
        fatal_error("Spectrum File: "+spec_file+" does not exist at the specified location")
    mx = np.unique(specData[mCol])
    xLog = np.unique(specData[xCol])
    dnData = specData[nCol]
    #dnData.reshape((len(mx),len(xLog)))
    if mode == "annihilation":
        intp = interp2d(mx,xLog,dnData,fill_value=0.0)
    else:
        intp = interp2d(mx,xLog,dnData,fill_value=0.0)
    return intp    

def readInputFile(inputFile):
    """
    Reads a yaml file and builds dictionaries 

    Arguments
    ---------------------------
    inputFile : str 
        Path of input file

    Returns
    ---------------------------
    datasets : dictionaries
        Dictionaries storing information on: calculations, halo properties, particle physics, magnetic fields, gas distribution, diffusion, and cosmology

    Notes
    ---------------------------
    All dictionaries are returned, empty dictionaries indicate no properties were set in the file
    """
    stream = open(inputFile, 'r')
    inputData = yaml.load(stream,Loader=yaml.SafeLoader)
    validKeys = ["haloData","magData","gasData","diffData","partData","calcData","cosmoData"]
    dmUnits = {"distance":"Mpc","mass":"Msun","density":"Msun/Mpc^3","numDensity":"1/cm^3","magnetic":"microGauss","energy":"GeV","frequency":"MHz","angle":"arcmin","jFactor":"GeV^2/cm^5","dFactor":"GeV/cm^2","diffConstant":"cm^2/s"}
    dataSets = {}
    for key in validKeys:
        dataSets[key] = {}
    for h in inputData.keys():
        if not h in validKeys:
            fatal_error("The key {} in the file {} is not valid, options are {}".format(h,inputFile,validKeys))
        for x in inputData[h].keys():
            if not isinstance(inputData[h][x],dict):
                dataSets[h][x] = inputData[h][x]
            elif 'unit' in inputData[h][x].keys():
                quant = checkQuant(x) #we find out what kind of units x has, i.e. distance, mass etc
                if not quant is None:
                    unitStr = dmUnits[quant] #get the unit DM uses internally
                else:
                    fatal_error("{} property {} does not accept a unit argument".format(h,x))
                try:
                    dataSets[h][x] = (inputData[h][x]['value']*units.Unit(inputData[h][x]['unit'])).to(unitStr).value #convert the units to internal system
                except:
                    fatal_error("Processing failed on {} property {} ".format(h,x))
    if len(dataSets['magData']) > 0:
        dataSets['magData']['magFuncLock'] = False
    if len(dataSets['calcData']) > 0:
        dataSets['calcData']['results'] = {'electronData':[],'radioEmData':[],'primaryEmData':[],'secondaryEmData':[],'finalData':[],'neutrinoEmData':[]}
    return dataSets

def readDMOutput(fName):
    """
    Reads in an output yaml file created by DarkMatters

    Arguments
    ---------------------------
    fName : str 
        Path of file

    Returns
    ---------------------------
    Dictionaries storing information on: calculations, halo properties, particle physics, magnetic fields, gas distribution, diffusion, and cosmology
    """
    stream = open(fName, 'r')
    inData = yaml.load(stream,Loader=yaml.UnsafeLoader)
    return inData['calcData'],inData['haloData'],inData['partData'],inData['magData'],inData['gasData'],inData['diffData'],inData['cosmoData']
