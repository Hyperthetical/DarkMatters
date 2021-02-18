#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:26:41 2019

@author: Geoff Beck
"""


from numpy import *
import sys

"""
Constants
"""
kb = 1.38064852e-23  #units =  m2 kg s-2 K-1
c = 3.0e+10  #units = cm s-1
Me = 0.511e-3  #units = GeV


def eloss_vector(E_vec,B,ne,z):
    #vectorised energy loss calculation
    #calculates b(E) for every value in E_set
    #E_vec is the set of energies, B is the mean magnetic field, ne is the mean plasma density, z is redshift
    n = ne#*(1+z)**3
    me = 0.511e-3 #GeV -> E comes in as E/me
    eloss = 6.08e-16*(me*E_vec)**2*(1+z)**4 + 0.0254e-16*(me*E_vec)**2*B**2 + 6.13e-16*n*(1+log(E_vec/n)/75.0)+ 1.51e-16*n*(0.36+log(E_vec/n))
    return where(ne==0.0,6.08e-16*(me*E_vec)**2*(1+z)**4 + 0.0254e-16*(me*E_vec)**2*B**2,eloss)

def D(E_set,B,lc,delta):
    #this is the diffusion coefficient
    #E_set is the energy domain, B is the mean magnetic field
    #lc is the minimum homogenuity scale for the field
    #delta is the turbulence spectral index
    me = 0.511e-3
    d0= (1.0/3.0)*(3e8)**(delta-1.0)*(1.6e-19)**(delta-2.0)*(1.6e-10)**(2.0-delta)
    d0 = d0*(B*1e-10)**(delta-2.0)*(lc*3.09e19)**(delta-1.0)*1e4   #cm^2 s^-1
    d0 = exp(log(d0)-2.0*log(3.09e24))   #Mpc^2 s^-1
    dset = d0*(E_set*me)**(2.0-delta)
    return dset


def dNdr(dndE,sim):
    """
    Part of the convection term in the diffusion equation, setup to evolve in time when
    used with Runge-Kutta.We start with an array of zeros. Then from our initial
    values for dndE we step along the r-coord. E is the row coord, r is the 
    column coord
    """
    
    delta_r = (sim['rf']-sim['r0'])/sim['nr']
    #zeros with sim['nE'] rows and sim['nr'] columns
    dndr =  zeros((sim['nE'],sim['nr']))
    
    for i in range(0,sim['nE']):
        for j in range(0,sim['nr']):
            if j < 2:
                #forward:
                dndr[i][j] = (-25.0/12 * dndE[i][j] + 4 * dndE[i][j+1] - 3 * dndE[i][j+2] + 4.0/3 * dndE[i][j+3] -0.25 * dndE[i][j+4])/delta_r
            elif j >= sim['nr'] - 2:
                #backward
                dndr[i][j] = (0.25 * dndE[i][j-4] - 4.0/3 * dndE[i][j-3] + 3.0 * dndE[i][j-2] - 4.0 * dndE[i][j-1] + 25/12 * dndE[i][j])/delta_r
            else:
                #central
               dndr[i][j] =  (1/12 * dndE[i][j-2] - 2/3 *  dndE[i][j-1] + 2/3 * dndE[i][j+1] -1/12 * dndE[i][j+2])/delta_r               
    return dndr

def LogdNdr(dndE,sim):
    """
    The dNdr term in log10 spacing, because we are working in vastly different 
    order of magnitude distances, i.e. cm and pc. Furthermore, here we use central 
    finite differences, we do not use boundary conditions per se, but rather use
    selective finite differrencing: i.e. we use central diff for all points except 
    final and start. For the start we use forward diff and for the final point we
    use backwards diff, will specify actual log10r values below when doing calc-
    ulations. THIS METHOD TAKES IN THE LOG DERIVATIVE AND REUTRNS THE ACTUAL 
    DERIVATIVE
    """
    
    #log10 delta_r:
    delta_r = log10((sim['rf']/sim['r0']))/sim['nr']
    Logdndr =  zeros((sim['nE'],sim['nr']))
    
    for i in range(0,sim['nE']):
        for j in range(0,sim['nr']):
            #actual r:
            r = (sim['rf']/sim['r0'])**(j/sim['nr'])*sim['r0']
            if j == 0*sim['nr']:
                #divide by r to convert back to ordinary derivatives from the log derivatives:
                Logdndr[i][j] = (-1.5 * dndE[i][j] + 2 * dndE[i][j+1] - 0.5 * dndE[i][j+2])/delta_r/r 
            elif j == sim['nr'] - 1:
                Logdndr[i][j] = (0.5 * dndE[i][j-2] - 2 * dndE[i][j-1] + 1.5 * dndE[i][j])/delta_r/r             
            else:
                Logdndr[i][j] = (0.5 * dndE[i][j+1] - 0.5 * dndE[i][j-1])/delta_r/r
            #constant to complete conversion mentioned above:
    return Logdndr*log10(exp(1.0))

def LogdNdrO4(dndE,sim):
    """
    The dNdr term in log10 spacing to 4th order accuracy.
    THIS METHOD TAKES IN THE LOG DERIVATIVE AND REUTRNS THE ACTUAL 
    DERIVATIVE. See LogdNdr for the 2nd order version
    """
    
    #log10 delta_r:
    delta_r = log10((sim['rf']/sim['r0']))/sim['nr']
    Logdndr =  zeros((sim['nE'],sim['nr']))
    
    for i in range(0,sim['nE']):
        for j in range(0,sim['nr']):
            #actual r:
            r = (sim['rf']/sim['r0'])**(j/sim['nr'])*sim['r0']
            if j < 2:
                #divide by r to convert back to ordinary derivatives from the log derivatives:
                Logdndr[i][j] = (-25/12 * dndE[i][j] + 4 * dndE[i][j+1] - 3 * dndE[i][j+2] + 4.0/3 * dndE[i][j+3] -0.25 * dndE[i][j+4])/delta_r/r 
            elif j >= sim['nr'] - 2:
                Logdndr[i][j] = (0.25 * dndE[i][j-4] -4.0/3 * dndE[i][j-3] + 3 * dndE[i][j-2] - 4  * dndE[i][j-1] + 25.0/12 * dndE[i][j])/delta_r/r             
            else:
                Logdndr[i][j] = (1.0/12 * dndE[i][j-2] - 2.0/3 *  dndE[i][j-1] + 2.0/3 * dndE[i][j+1] -1.0/12 * dndE[i][j+2])/delta_r/r
            #constant to complete conversion mentioned above:
    return Logdndr*log10(exp(1.0))

def LogdNdr_O4_vec(dndE,sim):
    """
    The dNdr term in log10 spacing to 4th order accuracy.
    THIS METHOD TAKES IN THE LOG-SPACED DISTRIBUTION AND REUTRNS THE ACTUAL 
    DERIVATIVE LOG-SPACED. See LogdNdr for the 2nd order version
    """
    
    #log10 delta_r:
    delta_r = log10((sim['rf']/sim['r0']))/sim['nr']
    Logdndr =  zeros((sim['nE'],sim['nr']))
    
    r = logspace(log10(sim['r0']),log10(sim['rf']),num=sim['nr'])
    #below we only take the r-elements from the third to third from last. this 
    #because for central differences you need two points surrounding any point
    #considered (for 4th order), which cannot be the case for i=1,2,n,n-1 etc. 
    #So we start with [2:-2]:
    rgrid = tensordot(ones(sim['nE']),r[2:-2],axes=0)
    #central coefficients:
    coeff_c = [1.0/12,-2.0/3,2.0/3,-1.0/12]
    #forward coefficients, backward coefficients found by reversing this array:
    coeff_f = [-25.0/12,4.0,-3.0,4.0/3,-0.25]
    
    Logdndr[:,2:-2] = (coeff_c[0]*dndE[:,:-4] + coeff_c[1]*dndE[:,1:-3] + coeff_c[2]*dndE[:,3:-1] + coeff_c[3]*dndE[:,4:])/delta_r/rgrid
    Logdndr[:,:2] = (coeff_f[0]*dndE[:,:2] + coeff_f[1]*dndE[:,1:3] + coeff_f[2]*dndE[:,2:4] + coeff_f[3]*dndE[:,3:5] + coeff_f[4]*dndE[:,4:6])/delta_r/tensordot(ones(sim['nE']),r[:2],axes=0)
    Logdndr[:,-2:] = (-coeff_f[0]*dndE[:,-2:] - coeff_f[1]*dndE[:,-3:-1] - coeff_f[2]*dndE[:,-4:-2] - coeff_f[3]*dndE[:,-5:-3] - coeff_f[4]*dndE[:,-6:-4])/delta_r/tensordot(ones(sim['nE']),r[-2:],axes=0)
    #constant to complete conversion mentioned above:
    return Logdndr*log10(exp(1.0))

def LogdNdE_O4_vec(dndE,sim):
    """
    The dNdr term in log10 spacing to 4th order accuracy.
    THIS METHOD TAKES IN THE LOG-SPACED DISTRIBUTION AND REUTRNS THE ACTUAL 
    DERIVATIVE LOG-SPACED. See LogdNdr for the 2nd order version
    """
    
    #log10 delta_r:
    delta_E = log10((sim['Ef']/sim['E0']))/sim['nE']
    LogdndE =  zeros((sim['nE'],sim['nr']))
    me = 0.511e-3 #GeV
    
    E = logspace(log10(sim['E0']),log10(sim['Ef']),num=sim['nE'])
    #below we only take the r-elements from the third to third from last. this 
    #because for central differences you need two points surrounding any point
    #considered (for 4th order), which cannot be the case for i=1,2,n,n-1 etc. 
    #So we start with [2:-2]:
    Egrid = tensordot(E[2:-2]*me,ones(sim['nr']),axes=0)
    #central coefficients:
    coeff_c = [1.0/12,-2.0/3,2.0/3,-1.0/12]
    #forward coefficients, backward coefficients found by reversing this array:
    coeff_f = [-25.0/12,4.0,-3.0,4.0/3,-0.25]
    
    LogdndE[2:-2,:] = (coeff_c[0]*dndE[:-4,:] + coeff_c[1]*dndE[1:-3,:] + coeff_c[2]*dndE[3:-1,:] + coeff_c[3]*dndE[4:,:])/delta_E/Egrid
    LogdndE[:2,:] = (coeff_f[0]*dndE[:2,:] + coeff_f[1]*dndE[1:3,:] + coeff_f[2]*dndE[2:4,:] + coeff_f[3]*dndE[3:5,:] + coeff_f[4]*dndE[4:6,:])/delta_E/tensordot(E[:2],ones(sim['nr']),axes=0)
    LogdndE[-2:,:] = (-coeff_f[0]*dndE[-2:,:] - coeff_f[1]*dndE[-3:-1,:] - coeff_f[2]*dndE[-4:-2,:] - coeff_f[3]*dndE[-5:-3,:] - coeff_f[4]*dndE[-6:-4,:])/delta_E/tensordot(E[-2:],ones(sim['nr']),axes=0)
    #constant to complete conversion mentioned above:
    return LogdndE*log10(exp(1.0))
                    
                

def dN2dr2(dndE,sim):
    """
    The second part in the diffusion part of the overall diffusion equation. 
    Again, we start with an array of zeros and must use periodic
    boundary conditions.
    """
    
    delta_r = (sim['rf']-sim['r0'])/sim['nr']
    dn2dr2 =  zeros((sim['nE'],sim['nr']))
    
    for i in range(0,sim['nE']):
        for j in range(0,sim['nr']):
            if j + 3 < sim['nr']:
                dn2dr2[i][j] = (2 * dndE[i][j] - 5 * dndE[i][j+1] + 4 * dndE[i][j+2] - dndE[i][j+3])/delta_r
            else:
                jp1 = j + 1
                jp2 = j + 2
                jp3 = j + 3
                #the greatest value j +1 could be is nr:
                if jp1 >= sim['nr']:
                    #this says that if we are outside of the boundary, then we 
                    #take one step back to the point mirrored to it, similarly 
                    #for j+2 = two steps back, etc:
                    jp1 = j - (j+1-sim['nr'])
                if jp2 >= sim['nr']:
                    jp2 = j - (j+2-sim['nr'])
                if jp3 >= sim['nr']:
                    jp3 = j - (j+3-sim['nr'])
                dn2dr2[i][j] = (-11/6 * dndE[i][j] + 3 * dndE[i][jp1] - 3/2 * dndE[i][jp2] + 1/3 * dndE[i][jp3])/delta_r
    return dn2dr2
                   
def Logd2Ndr2_O4_vec(dndE,sim): 
    delta_r = log10((sim['rf']/sim['r0']))/sim['nr']
    Logdndr =  zeros((sim['nE'],sim['nr']))
    
    r = logspace(log10(sim['r0']),log10(sim['rf']),num=sim['nr'])
    #below we only take the r-elements from the third to third from last. this 
    #because for central differences you need two points surrounding any point
    #considered (for 4th order), which cannot be the case for i=1,2,n,n-1 etc. 
    #So we start with [2:-2]:
    rgrid = tensordot(ones(sim['nE']),r[2:-2],axes=0)
    #central coefficients:
    coeff_c = [-1.0/12,4.0/3,-5.0/2,4.0/3,-1.0/12]
    #forward coefficients, backward coefficients found by reversing this array:
    coeff_f = [15.0/4,-77.0/6,107.0/6,-13,61.0/12,-5.0/6]
    
    Logdndr[:,2:-2] = (coeff_c[0]*dndE[:,:-4] + coeff_c[1]*dndE[:,1:-3] + coeff_c[2]*dndE[:,2:-2] + coeff_c[3]*dndE[:,3:-1] + coeff_c[4]*dndE[:,4:])/delta_r**2/rgrid**2
    Logdndr[:,:2] = (coeff_f[0]*dndE[:,:2] + coeff_f[1]*dndE[:,1:3] + coeff_f[2]*dndE[:,2:4] + coeff_f[3]*dndE[:,3:5] + coeff_f[4]*dndE[:,4:6] + coeff_f[5]*dndE[:,5:7])/delta_r**2/tensordot(ones(sim['nE']),r[:2],axes=0)**2
    Logdndr[:,-2:] = (-coeff_f[0]*dndE[:,-2:] - coeff_f[1]*dndE[:,-3:-1] - coeff_f[2]*dndE[:,-4:-2] - coeff_f[3]*dndE[:,-5:-3] - coeff_f[4]*dndE[:,-6:-4] - coeff_f[5]*dndE[:,-7:-5])/delta_r**2/tensordot(ones(sim['nE']),r[-2:],axes=0)**2
    #constant to complete conversion mentioned above:
    return Logdndr*log10(exp(1.0))**2

def dndt_vec(dnde,sim,lossTable,diffTable,sourceTable):
    E_vec = logspace(log10(sim['E0']),log10(sim['Ef']),num=sim['nE'])
    r_vec = logspace(log10(sim['r0']),log10(sim['rf']),num=sim['nr'])
    Egrid = tensordot(E_vec,ones(sim['nr']),axes=0)*0.511e-3
    rgrid = tensordot(ones(sim['nE']),r_vec,axes=0)
    dndt = lossTable*LogdNdE_O4_vec(dnde,sim) + dnde*LogdNdE_O4_vec(lossTable,sim) 
    dndr = LogdNdr_O4_vec(dnde,sim)
    #print(dndt)
    dndt += 2*diffTable/rgrid*dndr + LogdNdr_O4_vec(dnde,sim)*LogdNdr_O4_vec(diffTable,sim)
    dndt += diffTable*(-1.0/rgrid*dndr + Logd2Ndr2_O4_vec(dnde,sim)) + sourceTable
    return dndt

def solveDiffusion(r_vec,E_vec,lossTable,diffTable,sourceTable,dt,tend):
    sim = {'r0':r_vec[0],'rf':r_vec[-1],'E0':E_vec[0],'Ef':E_vec[-1],'nE':len(E_vec),'nr':len(r_vec)}
    print(sim)
    dnde = zeros((sim['nE'],sim['nr']))
    diff = 1e3
    tol = 1e-1
    t = 0.0;n=0
    while diff > tol and t<tend:
        k1 = dndt_vec(dnde,sim,lossTable,diffTable,sourceTable)
#        if n <= 2 :
#            print(k1[:,0])
#        else:
#            sys.exit(2)
        diff = abs(1.0-sqrt(tensordot(abs(k1/dnde),abs(k1/dnde),axes=2))/(sim['nE']*sim['nr']))
        if isnan(diff):
            diff = 1e3
        #print(diff)
        if diff <= tol and (not isnan(diff)):
            return dnde,True
        k2 = dndt_vec(dnde+0.5*k1,sim,lossTable,diffTable,sourceTable)
        k3 = dndt_vec(dnde+0.5*k2,sim,lossTable,diffTable,sourceTable)
        k4 = dndt_vec(dnde+k3,sim,lossTable,diffTable,sourceTable)
        dnde = dnde + (k1 + 2*k2 + 2*k3 + k4)/6*dt
        t += dt
        n += 1
        print(dnde[:,0])
    print(diff,n,(not isnan(diff)),diff>tol)
    if t < tend:
        return dnde,True
    else:
        return dnde,False
        
def getElectrons_numeric(halo,phys): 
    lossTable = []
    r_index = 0
    for g in phys.spectrum[0]:
        lossTable.append(eloss_vector(g,halo.b_sample,halo.ne_sample,halo.z))
    lossTable = array(lossTable)
    diffTable = []
    for g in phys.spectrum[0]:
        diffTable.append(D(g,halo.b_sample,phys.lc,phys.delta))
    diffTable = array(diffTable)
    nwimp0 = sqrt(1.458e-33)**halo.mode_exp/halo.mode_exp*(1.0/phys.mx)**halo.mode_exp
    rhodm = nwimp0*halo.rho_dm_sample[r_index]
    sigV = 3e-26
    sourceTable = tensordot(phys.spectrum[1]/phys.me,rhodm,axes=0)*sigV
    dt = 1e-2;runs=0
    cmTompc = 1.0#3.24078e-25
    tol = 1e-1
    tend = 50*3e16
    dnde,flag = solveDiffusion(halo.r_sample[r_index],phys.spectrum[0],lossTable,diffTable,sourceTable,dt,tend)
    return 2*dnde
    
def getElectrons_numeric_final(halo,phys):
    lossTable = []
    r_index = 0
    for g in phys.spectrum[0]:
        lossTable.append(eloss_vector(g,halo.b_sample,halo.ne_sample,halo.z))
    lossTable = array(lossTable)
    diffTable = []
    for g in phys.spectrum[0]:
        diffTable.append(D(g,halo.b_sample,phys.lc,phys.delta))
    diffTable = array(diffTable)
    nwimp0 = sqrt(1.458e-33)**halo.mode_exp/halo.mode_exp*(1.0/phys.mx)**halo.mode_exp
    rhodm = nwimp0*halo.rho_dm_sample[r_index]
    sourceTable = tensordot(phys.spectrum[1]/phys.me,rhodm,axes=0)
    dt = 1.0;runs=0
    cmTompc = 1.0#3.24078e-25
    tol = 1e-1
    tend = 50*3e16
    while runs < 1:
        dnde,flag = solveDiffusion(halo.r_sample[r_index],phys.spectrum[0],lossTable,diffTable,sourceTable,dt,tend)
        dnde2,flag2 = solveDiffusion(halo.r_sample[r_index],phys.spectrum[0],lossTable,diffTable,sourceTable,dt*0.5,tend)
        diff = abs(dnde/dnde2)
        print(flag,flag2)
        runs += 1
        diff = abs(sum(diff)/len(phys.spectrum[0])/len(halo.r_sample[r_index])-1.0)
        print(diff,runs)
        if (not isnan(diff)) and diff <= tol:
            return dnde
        else:
            dt = 0.5*dt
    return 2*dnde
    
