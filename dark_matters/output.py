import numpy as np
import sys
import pickle,yaml,json
import os
from scipy.interpolate import interp2d,interp1d
from astropy import wcs
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits


def checkQuant(key):
    inFile =open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"config/quantities.yaml"),"r")
    quantDict = yaml.load(inFile,Loader=yaml.SafeLoader)
    inFile.close()
    for h in quantDict:
        if key in quantDict[h]:
            return h
    else:
        return None

def fatal_error(err_string):
    """
    Display error string and exit program
        ---------------------------
        Parameters
        ---------------------------
        err_sting - Required : error message (String)
        ---------------------------
        Output
        ---------------------------
        None
    """
    print("################################################")
    print("                   Fatal Error")
    print("################################################")
    print(err_string)
    sys.exit(2)

def getCalcID(calcData,haloData,partData):
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
    dm_str = haloData['haloProfile']
    if 'haloIndex' in haloData.keys():
        dm_str += "-"+haloData['haloIndex']
    dm_str += "_"

    if not calcData['calcMode'] == "jflux":
        if haloData['haloWeights'] == "flat":
            w_str = "weights-flat_"
        else:
            w_str = "weights-rho_"

        if haloData['haloDistance'] < 0.1 or haloData['haloDistance'] > 1e3:
            dist_str = "dl-{:.2e}Mpc_".format(haloData['haloDistance'])
        elif  haloData['haloDistance'] == "0":
            dist_str = ""
        else:
            dist_str = "dl-{:.1f}Mpc_".format(haloData['haloDistance'])
    else:
        w_str = ""
        if partData['emModel'] == "annihilation":
            dist_str = "jfactor-{:.1e}_".format(haloData['haloJFactor'])
        else:
            dist_str = "dfactor-{:.1e}_".format(haloData['haloDFactor'])
    
    if partData['emModel'] == "decay":
        wimp_str = "decay_"
    else:
        wimp_str = "annihilation_"

    mxStr = "mx-"
    for mx in calcData['mWIMP']:
        mxStr += "{}".format(int(mx))
        if not mx == calcData['mWIMP'][-1]:
            mxStr += "-"
        else:
            mxStr += "GeV_"

    model_str = partData['partModel']+"_"
    return haloData['haloName']+"_"+model_str+mxStr+wimp_str+dm_str+w_str+dist_str

def fluxLabel(calcData):
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

def makeOutput(calcData,haloData,partData,magData,gasData,diffData,cosmoData,outMode="yaml",fName=None,emOnly=False):
    if np.any(calcData['results']['finalData'] is None):
        fatal_error("output.fitsMap() cannot be invoked without a full set of calculated results, some masses have not had calculations run")
    def default(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        raise TypeError('Unknown type:', type(obj))
    writeHalo = {key: value for key, value in haloData.items() if not key == 'haloDensityFunc'}
    writeMag = {key: value for key, value in magData.items() if not key == 'magFieldFunc'}
    writeGas = {key: value for key, value in gasData.items() if not key == 'gasDensityFunc'}
    writePart = {key: value for key, value in partData.items() if not key == 'dNdxInterp'}
    writeCalc = {key: value for key, value in calcData.items() if not key == 'results'}
    if emOnly:
        writeCalc['results'] = {key: value for key, value in calcData['results'].items() if not key == 'finalData'}
        if calcData['calcMode'] == "jflux":
            print("Warning from output.makeOutput(): jflux calculations have no emissivity, emOnly = True cannot be used!, reverting to emOnly = False")
            writeCalc['results'] = {key: value for key, value in calcData['results'].items()}
            emOnly = False
    else:
        writeCalc['results'] = {key: value for key, value in calcData['results'].items()}
    outData = {'calcData':writeCalc,'haloData':writeHalo,'partData':writePart,'magData':writeMag,'gasData':writeGas,'diffData':diffData,'cosmoData':cosmoData}
    if fName is None:
        fName = getCalcID(calcData,haloData,partData)+fluxLabel(calcData)
    if not emOnly:
        fName += "_"+calcData['calcMode']
    else:
        fName += "_emissivity"
    if outMode == "yaml":
        outFile = open(fName+".yaml","w")
        yaml.dump(outData, outFile)
        outFile.close()
    elif outMode == "pickle":
        outFile = open(fName+".pkl","w")
        pickle.dump(outData,outFile)
        outFile.close()
    elif outMode == "json":
        outFile = open(fName+".json","w")
        json.dump(outData,outFile,default=default)
        outFile.close()

def wimpWrite(mx,partData,target=None):
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
    outstream.write(prefix+'======================================================'+end)
    outstream.write(prefix+'Dark Matter Parameters: '+end)
    outstream.write(prefix+'======================================================'+end)
    outstream.write(prefix+'WIMP mass: '+str(mx)+' GeV'+end)
    outstream.write(prefix+'Dark matter particle physics: '+str(partData['partModel'])+end)
    outstream.write(prefix+"Emission type: "+str(partData['emModel'])+end)

def calcWrite(calcData,haloData,partData,magData,gasData,diffData,target=None):
    """
    Write calculation data to a target output
        ---------------------------
        Parameters
        ---------------------------
        log       - Optional : log file name (if None uses stdout) (String or None)
        writeMode - Optional : 'flux' displays all information (String)
        fluxMode  - Optional : can be used to exclude irrelevant Bfield and gas info (jflux,gflux,nuflux,nu_jflux) (String)
        ---------------------------
        Output
        ---------------------------
        Writes to a file or stdout
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
    outstream.write(prefix+'======================================================'+end)
    outstream.write(prefix+'Run Parameters'+end)
    outstream.write(prefix+'======================================================'+end)
    if 'calcOutputDirectory' in calcData.keys():
        outstream.write(prefix+"Output directory: "+calcData['calcOutputDirectory']+end)
    #outstream.write(prefix+'Field File Code: b'+str(int(phys.b0))+"q"+str(phys.qb)+end)
    outstream.write((prefix+"Frequency Samples: {}"+end).format(calcData['fSampleNum']))
    outstream.write((prefix+"Minimum Frequency Sampled: {:.1e} MHz"+end).format(calcData['fSampleLimits'][0]))
    outstream.write((prefix+"Maximum Frequency Sampled: {:.1e} MHz"+end).format(calcData['fSampleLimits'][1]))
    if not calcData['calcMode'] == "jflux":
        outstream.write(prefix+"Radial Grid Intervals: "+str(calcData['rSampleNum'])+end)
        if calcData['freqMode'] in ['all','radio','sgamma']:
            outstream.write((prefix+"Green's Function Grid Intervals: {}"+end).format(calcData['rGreenSampleNum']))
        outstream.write((prefix+'Minimum Sampled Radius: {:.2e} Mpc'+end).format(haloData['haloScale']*10**calcData['log10RSampleMinFactor']))
        outstream.write((prefix+'Maximum Sampled Radius: {:.2e} Mpc'+end).format(haloData['haloRvir']*2))
    outstream.write(prefix+'======================================================'+end)
    outstream.write(prefix+'Halo Parameters: '+end)
    outstream.write(prefix+'======================================================'+end)
    outstream.write(prefix+'Halo Name: '+str(haloData['haloName'])+end)
    if not calcData['calcMode'] == "jflux":
        outstream.write((prefix+'Redshift z: {:.2e}'+end).format(haloData['haloZ']))
        outstream.write((prefix+'Luminosity Distance: {:.2f} Mpc'+end).format(haloData['haloDistance']))
        outstream.write(prefix+'Halo profile: '+str(haloData['haloProfile'])+end)
        outstream.write((prefix+"Virial Mass: {:.2e} Solar Masses"+end).format(haloData['haloMvir']))
        outstream.write((prefix+'Virial Radius: {:.2e} Mpc'+end).format(haloData['haloRvir']))
        outstream.write((prefix+'Halo scale radius: {:.2e} Mpc'+end).format(haloData['haloScale']))
        outstream.write((prefix+'Rho_s/Rho_crit: {:.2e}'+end).format(haloData['haloNormRelative']))
        outstream.write((prefix+'Virial Concentration: {:.1f}'+end).format(haloData['haloCvir']))
    elif partData['emModel'] == "decay":
        outstream.write((prefix+'Dfactor: {:.2e} GeV cm^-2'+end).format(haloData['haloJFactor']))
    else:
        outstream.write((prefix+'Jfactor: {:.2e} GeV^2 cm^-5'+end).format(haloData['haloJFactor']))

    if calcData['freqMode'] in ["all","sgamma","radio"]:
        outstream.write(prefix+'======================================================'+end)
        outstream.write(prefix+'Gas Parameters: '+end)
        outstream.write(prefix+'======================================================'+end)
        outstream.write(prefix+"Gas density profile: {}".format(gasData['gasProfile'])+end)
        paramSet = gasParams[gasData['gasProfile']]
        for p in paramSet:
            unitType = checkQuant(p)
            if not unitType is None:
                outstream.write(prefix+"{}: {} {}".format(p,gasData[p],unitDict[unitType])+end)
            else:
                outstream.write(prefix+"{}: {}".format(p,gasData[p])+end)
        outstream.write(prefix+'======================================================'+end)
        outstream.write(prefix+'Magnetic Field Parameters: '+end)
        outstream.write(prefix+'======================================================'+end)
        outstream.write(prefix+"Magnetic field profile: {}".format(magData['magProfile'])+end)
        paramSet = magParams[magData['magProfile']]
        for p in paramSet:
            unitType = checkQuant(p)
            if not unitType is None:
                outstream.write(prefix+"{}: {} {}".format(p,magData[p],unitDict[unitType])+end)
            else:
                outstream.write(prefix+"{}: {}".format(p,magData[p])+end)
        if diffData['lossOnly']:
            outstream.write(prefix+'No Diffusion'+end)
        else:
            outstream.write(prefix+'Spatial Diffusion'+end)
            outstream.write((prefix+'Turbulence scale: {:.2e} kpc'+end).format(diffData['coherenceScale']*1e3))
            outstream.write((prefix+'Turbulence index: {:.2f}'+end).format(diffData['diffIndex']))
            outstream.write((prefix+'Diffusion constant: {:.2e} cm^2 s^-1'+end).format(diffData['diffConstant']))
    if stringOut:
        return outstream.text
    elif not target is None:
        outstream.close()

def fitsMap(skyCoords,targetFreqs,calcData,haloData,partData,sigV=1e-26,halfPix=3000,useHalfPix=500,display_slice=None):
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
    useStartPix = halfPix-useHalfPix
    useEndPix = useStartPix + 2*useHalfPix
    rSet = np.logspace(np.log10(haloData['haloScale']*10**calcData['log10RSampleMinFactor']),np.log10(haloData['haloRvir']*2),num=calcData['rSampleNum'])
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
            rMax = rSet[-1]/45
            rMin = rSet[0]
            angleAlpha = (rMax-rMin)/(n-1) #for a conversion from array index to angular values
            angleBeta = rMax - angleAlpha*(n-1)
            rPlot = angleAlpha*rPlot + angleBeta
            arcMinPerPixel = rMax*2/n
            sPlot = intp(rPlot*1.00000001)*sigV*arcMinPerPixel**2
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
        hdr['CUNIT3'] = 'Hz'
        hdr['CTYPE4'] = 'STOKES'
        hdr['CRPIX4'] = 1
        hdr['CRVAL4'] = 1
        hdr['CDELT4'] = 1
        hdr['CUNIT4'] = ''
        hduList.append(hdu)

    hduList = fits.HDUList(hduList)
    hduList.writeto(getCalcID(calcData,haloData,partData)+fluxLabel(calcData)+".fits",overwrite=True)
