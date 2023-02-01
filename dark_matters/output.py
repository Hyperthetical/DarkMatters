import numpy as np
import sys
import yaml,json
import os
from scipy.interpolate import interp2d,interp1d
from astropy import wcs
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits

spacer_length = 55


def checkQuant(key):
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
    inFile =open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"config/quantities.yaml"),"r")
    quantDict = yaml.load(inFile,Loader=yaml.SafeLoader)
    inFile.close()
    for h in quantDict:
        if key in quantDict[h]:
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

def getCalcID(calcData,haloData,partData,diffData,tag=None):
    """
    Builds an output file id code

    Arguments
    ---------------------------
    calcData : dictionary
        Calculation properties
    haloData : dictionary
        Halo properties
    partData : dictionary
        Particle physics properties
    diffData : dictionary
        Diffusion properties
    tag : string (optional)
        String added to file ID

    Returns
    ---------------------------
    file_id : string
        File name for calculations
    """
    dm_str = haloData['haloProfile']
    if 'haloIndex' in haloData.keys():
        dm_str += f"-{haloData['haloIndex']:.2f}"
    dm_str += "_"

    if not calcData['calcMode'] == "jflux":
        if not "green" in calcData['electronMode']:
            w_str = ""
        elif haloData['haloWeights'] == "flat":
            w_str = "weights-flat_"
        else:
            w_str = "weights-rho_"

        if haloData['haloDistance'] < 0.1 or haloData['haloDistance'] > 1e3:
            dist_str = f"dl-{haloData['haloDistance']:.2e}Mpc_"
        elif  haloData['haloDistance'] == "0":
            dist_str = ""
        else:
            dist_str = f"dl-{haloData['haloDistance']:.1f}Mpc_"
    else:
        w_str = ""
        if partData['emModel'] == "annihilation":
            dist_str = f"jfactor-{haloData['haloJFactor']:.1e}_"
        else:
            dist_str = f"dfactor-{haloData['haloDFactor']:.1e}_"

    if calcData['freqMode'] in ['gamma','sgamma','radio','all']:
        fm_str = calcData['electronMode']+"_"
    else:
        fm_str = ""
    
    if partData['emModel'] == "decay":
        wimp_str = "decay_"
    else:
        wimp_str = "annihilation_"

    mxStr = "mx-"
    for mx in calcData['mWIMP']:
        mxStr += f"{int(mx)}"
        if not mx == calcData['mWIMP'][-1]:
            mxStr += "-"
        else:
            mxStr += "GeV_"

    diff_str = ""
    if diffData['lossOnly']:
        if calcData['freqMode'] in ['gamma','sgamma','radio','all']:
            diff_str = "loss-only_"
    else:
        if calcData['freqMode'] in ['gamma','sgamma','radio','all']:
            diff_str = f"D0-{diffData['diffConstant']}_"

    model_str = partData['partModel']+"_"
    if not tag is None:
        tag_str = tag+"_"
    else:
        tag_str = ""
    return haloData['haloName']+"_"+model_str+mxStr+wimp_str+dm_str+fm_str+w_str+dist_str+diff_str+tag_str

def fluxLabel(calcData):
    """
    Find frequency label string for calculation
    
    Arguments
    ---------------------------
    calcData : dictionary
        Calculation properties

    Returns
    ---------------------------
    fluxStr : string
        Frequency label for output file
    """
    if calcData['freqMode'] == "radio":
        fluxStr = "sync"
    elif calcData['freqMode'] == "gamma":
        fluxStr = "gamma"
    elif calcData['freqMode'] == "pgamma":
        fluxStr = "primary_gamma"
    elif calcData['freqMode'] == "sgamma":
        fluxStr = "secondary_gamma"
    elif "neutrinos" in calcData['freqMode']:
        fluxStr = calcData['freqMode']
    else:
        fluxStr = "multi_frequency"
    return fluxStr

def processDict(d,exclude=None,exclude_recurse=False):
    new_d = {}
    if exclude_recurse:
        r_ex = exclude
    else:
        r_ex = None
    for key,value in d.items():
        if isinstance(value,dict):
            new_d[key] = processDict(value,r_ex)
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

def makeOutput(calcData,haloData,partData,magData,gasData,diffData,cosmoData,outMode="yaml",tag=None,emOnly=False,noNumpy=False):
    """
    Write calculation data to an output file
    
    Arguments
    ---------------------------
    calcData : dictionary
        Calculation properties
    haloData : dictionary
        Halo properties
    partData : dictionary
        Particle physics properties
    magData : dictionary
        Magnetic field properties
    gasData : dictionary
        Gas density properties
    diffData : dictionary
        Diffusion properties
    cosmoData : dictionary
        Cosmology properties
    outMode : string, optional
        Can be "yaml","json","pickle"
    tag : string, optional
        String added to file name
    emOnly: boolean, optional
        True means calcData['results']['finalData'] is not written
    noNumpy : boolean, optional
        True means all data converted to native python formats, False writes numpy arrays as is (yaml only) 

    Returns
    ---------------------------
    None
    """
    if np.any(calcData['results']['finalData'] is None):
        fatal_error("output.makeOutput() cannot be invoked without a full set of calculated results, some masses have not had calculations run")
    def default(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        raise TypeError('Unknown type:', type(obj))

    if noNumpy and outMode=="yaml":
        writeHalo = processDict(haloData,exclude=['haloDensityFunc'])
        writeGas = processDict(gasData,exclude=['gasDensityFunc'])
        writeMag = processDict(magData,exclude=['magFieldFunc'])
        writePart = processDict(partData,exclude=['dNdxInterp'])
        writeDiff = processDict(diffData)
        writeCalc = processDict(calcData)
        writeCosmo = processDict(cosmoData)
    else:
        writeHalo = {key: value for key, value in haloData.items() if not key == 'haloDensityFunc'}
        writeMag = {key: value for key, value in magData.items() if not key == 'magFieldFunc'}
        writeGas = {key: value for key, value in gasData.items() if not key == 'gasDensityFunc'}
        writePart = {key: value for key, value in partData.items() if not key == 'dNdxInterp'}
        writeCalc = {key: value for key, value in calcData.items()}
        writeCosmo = cosmoData
        writeDiff = diffData

    if emOnly and calcData['calcMode'] == "jflux":
        warning("output.makeOutput(): jflux calculations have no emissivity, emOnly = True cannot be used!, reverting to emOnly = False")
        emOnly = False
    if emOnly:
        writeCalc['results'].pop('finalData')
    outData = {'calcData':writeCalc,'haloData':writeHalo,'partData':writePart,'magData':writeMag,'gasData':writeGas,'diffData':writeDiff,'cosmoData':writeCosmo}
    fName = getCalcID(calcData,haloData,partData,diffData,tag=tag)+fluxLabel(calcData)
    if not emOnly:
        fName += "_"+calcData['calcMode']
    else:
        fName += "_emissivity"
    if outMode == "yaml":
        outFile = open(fName+".yaml","w")
        yaml.dump(outData, outFile)
        outFile.close()
    elif outMode == "json":
        outFile = open(fName+".json","w")
        json.dump(outData,outFile,default=default)
        outFile.close()

def wimpWrite(mx,partData,target=None):
    """
    Write WIMP parameters to a target output
    
    Arguments
    ---------------------------
    mx : float
        WIMP mass [GeV]
    partData : dictionary
        Particle physics properties
    target : string (optional)
        Output file or stream name, None goes to stdout

    Returns
    ---------------------------
    None
    """
    class stringStream:
        def __init__(self,string):
            self.text = string

        def write(self,string):
            self.text += string
    end = "\n"
    stringOut = False
    if(target is None):
        outstream = sys.stdout
    elif os.path.isfile(target):
        outstream = open(target,"w")
    else:
        outstream = stringStream(target)
        end = ""
        stringOut = True
    if target is None:
        prefix = ""
    else:
        prefix = "#"
    outstream.write(f"{prefix}{'='*spacer_length} {end}")
    outstream.write(f"{prefix}Now calculating for Dark Matter model: {end}")
    outstream.write(f"{prefix}{'='*spacer_length}{end}")
    outstream.write(f"{prefix}WIMP mass: {mx} GeV{end}")
    outstream.write(f"{prefix}Particle physics: {partData['partModel']}{end}")
    outstream.write(f"{prefix}Emission type: {partData['emModel']}{end}")

def calcWrite(calcData,haloData,partData,magData,gasData,diffData,target=None):
    """
    Write calculation data to a target output
    
    Arguments
    ---------------------------
    calcData : dictionary
        Calculation properties
    haloData : dictionary
        Halo properties
    partData : dictionary
        Particle physics properties
    magData : dictionary
        Magnetic field properties
    gasData : dictionary
        Gas density properties
    diffData : dictionary
        Diffusion properties
    target : string (optional)
        Output file or stream name, None goes to stdout

    Returns
    ---------------------------
    None
    """
    class stringStream:
        def __init__(self,string):
            self.text = string

        def write(self,string):
            self.text += string
    gasParams = yaml.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"config/gasDensityProfiles.yaml"),"r"),Loader=yaml.SafeLoader)
    magParams = yaml.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"config/magFieldProfiles.yaml"),"r"),Loader=yaml.SafeLoader)
    unitDict = {"distance":"Mpc","magnetic":"micro-Gauss","numDensity":"cm^-3"}
    end = "\n"
    stringOut = False
    if(target is None):
        outstream = sys.stdout
    elif os.path.isfile(target):
        outstream = open(target,"w")
    else:
        outstream = stringStream(target)
        end = ""
        stringOut = True
    if target is None:
        prefix = ""
    else:
        prefix = "#"
    outstream.write(f"{prefix}{'='*spacer_length}{end}")
    outstream.write(f"{prefix}Run Parameters{end}")
    outstream.write(f"{prefix}{'='*spacer_length}{end}")
    if 'calcOutputDirectory' in calcData.keys():
        outstream.write(f"{prefix}Output directory: {calcData['calcOutputDirectory']}{end}")
    #outstream.write(f"{prefix}Field File Code: b'+str(int(phys.b0))+"q"+str(phys.qb)+end)
    outstream.write((f"{prefix}Frequency Samples: {calcData['fSampleNum']}{end}"))
    outstream.write(f"{prefix}Minimum Frequency Sampled: {calcData['fSampleLimits'][0]:.2e} MHz {end}")
    outstream.write(f"{prefix}Maximum Frequency Sampled: {calcData['fSampleLimits'][1]:.2e} MHz{end}")
    if not calcData['calcMode'] == "jflux":
        outstream.write(f"{prefix}Radial Grid Intervals: {calcData['rSampleNum']}{end}")
        if calcData['freqMode'] in ['all','radio','sgamma'] and "green" in calcData['electronMode']:
            outstream.write(f"{prefix}Green's Function Grid Intervals: {calcData['rGreenSampleNum']}{end}")
        if diffData['diffRmax'] == "2*Rvir":
            rLimit = 2*haloData['haloRvir']
        else:
            rLimit = diffData['diffRmax']
        outstream.write(f"{prefix}Minimum Sampled Radius: {haloData['haloScale']*10**calcData['log10RSampleMinFactor']:.3e} Mpc{end}")
        outstream.write((f"{prefix}Maximum Sampled Radius: {rLimit:.3e} Mpc{end}"))
    outstream.write(f"{prefix}{'='*spacer_length}{end}")
    outstream.write(f"{prefix}Halo Parameters: {end}")
    outstream.write(f"{prefix}{'='*spacer_length}{end}")
    outstream.write(f"{prefix}Halo Name: {haloData['haloName']}{end}")
    if not calcData['calcMode'] == "jflux":
        outstream.write(f"{prefix}Redshift z: {haloData['haloZ']:.2e}{end}")
        outstream.write(f"{prefix}Luminosity Distance: {haloData['haloDistance']:.3f} Mpc{end}")
        outstream.write(f"{prefix}Halo profile: {haloData['haloProfile']}{end}")
        if haloData['haloProfile'] in ["einasto","gnfw","cgnfw"]:
            outstream.write(f"{prefix}Halo index parameter: {haloData['haloIndex']}{end}")
        outstream.write(f"{prefix}Virial Mass: {haloData['haloMvir']:.3e} Solar Masses{end}")
        outstream.write(f"{prefix}Virial Radius: {haloData['haloRvir']:.3e} Mpc{end}")
        outstream.write(f"{prefix}Halo scale radius: {haloData['haloScale']:.3e} Mpc{end}")
        outstream.write(f"{prefix}Rho_s/Rho_crit: {haloData['haloNormRelative']:.3e}{end}")
        outstream.write(f"{prefix}Virial Concentration: {haloData['haloCvir']:.2f}{end}")
    elif partData['emModel'] == "decay":
        outstream.write(f"{prefix}Dfactor: {haloData['haloJFactor']:.2e} GeV cm^-2{end}")
    else:
        outstream.write(f"{prefix}Jfactor: {haloData['haloJFactor']:.2e} GeV^2 cm^-5{end}")

    if calcData['freqMode'] in ["all","sgamma","radio"]:
        outstream.write(f"{prefix}{'='*spacer_length}{end}")
        outstream.write(f"{prefix}Gas Parameters: {end}")
        outstream.write(f"{prefix}{'='*spacer_length}{end}")
        outstream.write(f"{prefix}Gas density profile: {gasData['gasProfile']}{end}")
        paramSet = gasParams[gasData['gasProfile']]
        for p in paramSet:
            unitType = checkQuant(p)
            if not unitType is None:
                outstream.write(f"{prefix}{p}: {gasData[p]} {unitDict[unitType]} {end}")
            else:
                outstream.write(f"{prefix}{p}: {gasData[p]}{end}")
        outstream.write(f"{prefix}{'='*spacer_length}{end}")
        outstream.write(f"{prefix}Magnetic Field Parameters: {end}")
        outstream.write(f"{prefix}{'='*spacer_length}{end}")
        outstream.write(f"{prefix}Magnetic field profile: {magData['magProfile']}{end}")
        paramSet = magParams[magData['magProfile']]
        for p in paramSet:
            unitType = checkQuant(p)
            if not unitType is None:
                outstream.write(f"{prefix}{p}: {magData[p]} {unitDict[unitType]} {end}")
            else:
                outstream.write(f"{prefix}{p}: {magData[p]}{end}")
        if diffData['lossOnly']:
            outstream.write(f"{prefix}No Diffusion{end}")
        else:
            outstream.write(f"{prefix}Spatial Diffusion{end}")
            outstream.write(f"{prefix}Turbulence index: {diffData['diffIndex']:.2f}{end}")
            outstream.write(f"{prefix}Diffusion constant: {diffData['diffConstant']:.2e} cm^2 s^-1{end}")
    if stringOut:
        return outstream.text
    elif not target is None:
        outstream.close()

def fitsMap(skyCoords,targetFreqs,calcData,haloData,partData,diffData,sigV=1e-26,halfPix=3000,useHalfPix=500,display_slice=None,rMax=None):
    """
    Output a fits file with radio maps
    
    Arguments
    ---------------------------
    skyCoords : SkyCoord (astropy object)
        Sky coordinates object from astropy
    targetFreqs : list
        Frequencies for fits slices [MHz]
    calcData : dictionary
        Calculation properties
    haloData : dictionary
        Halo properties
    partData : dictionary
        Particle physics properties
    magData : dictionary
        Magnetic field properties
    gasData : dictionary
        Gas density properties
    diffData : dictionary
        Diffusion properties
    sigV : float
        Cross-section or decay rate [cm^3 s^-1 or s^-1]
    halfPix : int
        Half number of image pixels
    useHalfPix : int
        Number of pixels that are retained (reduces square-circle distortion)
    display_slice : int
        Index of frequency to display in a plot
    rMax : float
        Radial extent of fits map [Mpc]

    Returns
    ---------------------------
    None
    """
    if not calcData['calcMode'] == "sb":
        fatal_error("output.fitsMap() can only be run with surface brightness data")
    if np.any(calcData['results']['finalData'] is None):
        fatal_error("output.fitsMap() cannot be invoked without a full set of calculated results, some masses have not had calculations run")
    #we use more pixels than we want to discard outer ones with worse resolution distortion from ogrid
    if halfPix < useHalfPix:
        fatal_error("output.fitsMap() arguments must be such that useHalfPix <= halfPix")
    else:
        try:
            halfPix = int(halfPix)
            useHalfPix = int(useHalfPix)
        except:
            fatal_error("output.fitsMap() parameters halfPix and useHalfPix must be integers")
    if not display_slice is None:
        try:
            int(display_slice)
        except:
            fatal_error("output.fitsMap() parameter display_slice must be an integer that addresses an element of targetFreqs")
        if display_slice >= len(targetFreqs) or display_slice < 0 or display_slice != int(display_slice):
            fatal_error("output.fitsMap() parameter display_slice must be an integer that addresses an element of targetFreqs")
    if diffData['diffRmax'] == "2*Rvir":
        rLimit = 2*haloData['haloRvir']
    else:
        rLimit = diffData['diffRmax']
    if not rMax is None:
        if rMax > rLimit:
            fatal_error(f"Argument rMax cannot be greater than largest sampled r value {rLimit} Mpc")
    else:
        rMax = rLimit
    useStartPix = halfPix-useHalfPix
    useEndPix = useStartPix + 2*useHalfPix
    rSet = np.logspace(np.log10(haloData['haloScale']*10**calcData['log10RSampleMinFactor']),np.log10(rLimit),num=calcData['rSampleNum'])
    rSet = np.arctan(rSet/haloData['haloDistance'])/np.pi*180*60 #must be arcmins for the algorithm below
    fSet = calcData['fSampleValues']
    hduList = []
    for mx in calcData['mWIMP']:
        fitsOutSet = []
        massIndex = np.where(calcData['mWIMP']==mx)[0][0]
        fullDataIntp = interp2d(rSet,fSet,calcData['results']['finalData'][massIndex],bounds_error=False,fill_value=0.0)
        for i in range(len(targetFreqs)):
            rData = fullDataIntp(rSet,targetFreqs[i])
            intp = interp1d(rSet,rData)
            circle = np.ogrid[-halfPix:halfPix,-halfPix:halfPix]
            rPlot = np.sqrt(circle[0]**2  + circle[1]**2)
            n = circle[0].shape[0]
            #rMax = rSet[-1]
            rMin = rSet[0]
            angleAlpha = (rMax-rMin)/(n-1) #for a conversion from array index to angular values
            angleBeta = rMax - angleAlpha*(n-1)
            rPlot = angleAlpha*rPlot + angleBeta
            arcMinPerPixel = rMax*2/n
            sPlot = intp(rPlot*1.00000001)*sigV/1e-26*arcMinPerPixel**2
            raVal = skyCoords.ra.value*60 #arcmin
            decVal = skyCoords.dec.value*60 #arcmin
            if not display_slice is None:
                if i == display_slice:
                    fig = plt.gcf()
                    ax = fig.gca()
                    ax.set_aspect('equal')
                    im = plt.imshow(sPlot,cmap="RdYlBu",norm=LogNorm(vmin=np.min(sPlot),vmax=np.max(sPlot)),extent=[(rMax+raVal)/60,(-rMax+raVal)/60,(-rMax+decVal)/60,(rMax+decVal)/60])
                    plt.xlabel("RA--SIN (degrees)")
                    plt.ylabel("DEC--SIN (degrees)")
                    cbar = plt.colorbar(im)
                    cbar.set_label(r"I$(\nu)$ Jy/pixel")
                    plt.show()
            fitsOutSet.append(sPlot[useStartPix:useEndPix,useStartPix:useEndPix])
        fitsOutSet = np.array(fitsOutSet)

        angleAlpha = 2*rMax/(n-1)
        angleBeta = rMax - angleAlpha*(n-1)

        raSet = np.flipud((np.arange(n)*angleAlpha+angleBeta+raVal)/60) #ra declines to the right in RA---SIN
        decSet = (np.arange(n)*angleAlpha+angleBeta+decVal)/60 

        raDelta = np.sum(-raSet[0:-1]+raSet[1:])/(n-1)
        decDelta = np.sum(-decSet[0:-1]+decSet[1:])/(n-1)

        #creating a world coordinate system for our fits file
        wCoords = wcs.WCS(naxis=2)
        wCoords.wcs.crpix = [useHalfPix,useHalfPix]
        wCoords.wcs.crval = [raSet[useStartPix+useHalfPix],decSet[useStartPix+useHalfPix]]
        wCoords.wcs.ctype = ['RA---SIN','DEC--SIN']
        wCoords.wcs.cdelt = [raDelta,decDelta]

        hdr = wCoords.to_header()
        if massIndex == 0:
            hdu = fits.PrimaryHDU(fitsOutSet,header=hdr)
        else:
            hdu = fits.ImageHDU(fitsOutSet,header=hdr)
        hdr = hdu.header
        hdr['BUNIT'] = 'JY/PIXEL'
        hdr['BZERO'] = 0.0
        hdr['BSCALE'] = 1.0
        hdr['EQUINOX'] = 2000
        hdr['BTYPE'] = 'INTENSITY'
        hdr['ORIGIN'] = 'DARKMATTERS'
        hdr['OBSERVER'] = 'DARKMATTERS'
        hdr['OBJECT'] = haloData['haloName'].upper()
        hdr['CTYPE3'] = 'FREQ'
        hdr['CRPIX3'] = 1
        hdr['CRVAL3'] = targetFreqs[0]*1e6
        hdr['CDELT3'] = (targetFreqs[1] - targetFreqs[0])*1e6
        hdr['CUNIT3'] = 'HZ'
        hdr['CTYPE4'] = 'STOKES'
        hdr['CRPIX4'] = 1
        hdr['CRVAL4'] = 1
        hdr['CDELT4'] = 1
        hdr['CUNIT4'] = ''
        hdr['WIMPMASS'] = f'{mx}'
        hdr['MASSUNIT'] = "GEV"
        hduList.append(hdu)

    hduList = fits.HDUList(hduList)
    hduList.writeto(getCalcID(calcData,haloData,partData,diffData)+fluxLabel(calcData)+".fits",overwrite=True)
