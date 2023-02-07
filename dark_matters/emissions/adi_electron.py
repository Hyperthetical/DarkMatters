import time,warnings
import numpy as np
from scipy import sparse
from astropy import units, constants
from .progress_bar import progress
from ..output import warning,spacer_length


class adi_scheme:
    """
    ADI scheme class

    Arguments
    ---------------------------
    benchmark_flag : boolean
        Flag for strong convergence condition (testing only)
    const_Delta_t : boolean
        Flag for constant time-step (testing only)
    animation_flag : boolean
        Flag for producing animation showing evolution of solution (slow)

    Attributes
    ---------------------------
    effect : str
        Enabled effects ["loss","diffusion","all"]
    r_grid : array-like float (n)
        Radial samples [Mpc]
    r_bins : int
        Number of radial samples
    E_grid : array-like float (m)
        Electron energy samples [GeV^-1]
    E_bins : int
        Number of electron energy samples
    Q : array-like float (n,m)
        Source function   [GeV^-1 s^-1]
    D : array-like float (n,m)
        Diffusion function [cm^2 s^-1]
    dDdr : array-like float (n,m)
        Spatial derivative of diffusion function [cm s^-1]
    b : array-like float (n,m)
        Loss function [GeV s^-1]
    dbdE : array-like float (n,m)
        Energy derivative of loss function [s^-1]
    loss_constants : dictionary
        Dictionary of energy-loss coefficients [GeV s^-1]
    r0 : float
        Radial normalisation scale
    E0 : float
        Energy normalisation scale
    logr_grid : array-like float (n)
        Log10 of normalised radial samples
    Delta_r : float
        Log-spacing of normalised radial samples
    logE_grid : array-like float (m)
        Log10 of normalised energy samples
    Delta_E : float
        Log-spacing of normalised energy samples
    Delta_t : float
        Current time-step [s]
    Delta_ti : float
        Initial time-step [s]
    max_t_part : int
        Number of iterations between time-step reductions
    Delta_t_reduction : float
        Reduction factor for time-step after max_t_part iterations
    smallest_Delta_t : float
        Smallest time-step in use [s]
    loss_ts : array-like float (n,m)
        Energy-loss time-scale [s]
    diff_ts : array-like float (n,m)
        Diffusion time-scale [s]
    delta : float
        Diffusion power-spectrum index
    D0 : float
        Diffusion constant [cm^2 s^-1]
    benchmark_flag : boolean
        Flag for strong convergence condition (testing only)
    const_Delta_t : boolean
        Flag for constant time-step (testing only)
    animation_flag : boolean
        Flag for producing animation showing evolution of solution (slow)
    electrons : array-like float (n,m)
        Output electron equilibrium distribution [GeV cm^-3]
    solveElectrons : function
        Sets up grids and calls ADI solver
    set_D : function
        Builds diffusion function
    set_dDdr : function
        Builds spatial derivative of diffusion function
    set_b : function
        Builds energy-loss function
    r_prefactor : function
        Builds prefactor for spatial log-spaced solution
    E_prefactor : function
        Builds prefactor for energy log-spaced solution
    r_alpha1 : function
        First propagator coefficients in space 
    r_alpha2 : function
        Second propagator coefficients in space 
    r_alpha3 : function
        Third propagator coefficients in space 
    E_alpha1 : function
        First propagator coefficients in energy
    E_alpha2 : function
        Second propagator coefficients in energy 
    E_alpha3 : function
        Third propagator coefficients in energy 
    spmatrices_loss : function
        Builds sparse matrices for energy propagator
    spmatrices_diff : function
        Builds sparse matrices for spatial propagator
    adi2D : function
        Runs ADI solution
    """
    def __init__(self,benchmark_flag=False,const_Delta_t=False,animation_flag=False):
        self.effect = None      #which effects to include in the solution of the transport equation (in set {"loss","diffusion","all"})

        self.Q = None           #source function (2D np array, size r_bins x E_bins) [pc, GeV^-1]
        self.electrons = None   #equilibrium electron distribution (2D np array, size r_bins x E_bins) [pc, GeV]
  
        self.r_bins = None      #radius domain from halo radius array (float)
        self.E_bins = None      #energy domain from wimp particle spectrum (float)
        self.r_grid = None      #spatial grid (1D np array, size n_grid)        
        self.E_grid = None      #energy grid (1D np array, size n_grid)
        self.r0 = None          #radius scale value
        self.E0 = None          #energy scale value  
        self.logr_grid = None   #log-transformed spatial grid
        self.logE_grid = None   #log-transformed energy grid 
        
        self.D = None           #diffusion function, sampled at (r_grid,E_grid)
        self.dDdr = None        #derivative of diffusion function, sampled at (r_grid,E_grid)
        self.b = None           #energy loss function, sampled at (r_grid,E_grid)
        self.dbdE = None        #derivative of energy loss function, sampled at (r_grid,E_grid)
        self.loss_constants = None   #dictionary of constants for energy loss functions (dict) [GeV s^-1]

        self.Delta_r = None     #step size for space dimension after transform (dimensionless)
        self.Delta_E = None     #step size for energy dimension after transform (dimensionless)
        self.Delta_t = None     #step size for temporal dimension [s]
        self.Delta_ti = None    #initial value for Delta_t

        self.loss_ts = None     #loss timescale
        self.diff_ts = None     #diffusion timescale

        self.benchmark_flag = benchmark_flag    #flag for whether run should use benchmark convergence condition  
        self.const_Delta_t = const_Delta_t    #flag for using a constant step size or not. If False, Delta_t is reduced during method (accelerated method), if True Delta_t remains constant.
        self.smallest_Delta_t = None    #smallest value of Delta_t before final convergence (for accelerated timestep switching method)
        self.max_t_part = None    #maximum number of iterations for each value of Delta_t in accelerated method
        self.Delta_t_reduction = None    #factor by which to reduce Delta_t during timestep-switching in accelerated method
        
        self.animation_flag = animation_flag      #flag for whether animations take place or not
        self.snapshots = None   #stores snapshots of psi and Delta_t at each iteration for animation
        
    def solveElectrons(self,mx,z,E_sample,r_sample,rho_sample,Q_sample,b_sample,dBdr_sample,ne_sample,rScale,eScale,delta,diff0=3.1e28,uPh=0.0,lossOnly=False,mode_exp=2,Delta_t_min=1e1,Delta_ti=1e9,max_t_part=100,Delta_t_reduction=0.5):
        """
        Set up and solve for electron distribution

        Arguments
        ------------------------
        mx : float
            WIMP mass [GeV]
        z : float
            Redshift of halo
        E_sample : array-like float (k)
            Yield function Lrentz-gamma values
        Q_sample : array-like float (k)
            (Yield function * electron mass) [particles per annihilation]
        r_sample : array-like float (n)
            Sampled radii [Mpc]
        rho_dm_sample : array-like float (n)
            Dark matter density at r_sample [Msun/Mpc^3]
        b_sample : array-like float (n)
            Magnetic field strength at r_sample [uG]
        dBdr_sample : array-like float (n)
            Magnetic field strength  derivative at r_sample [uG Mpc^-1]
        ne_sample : array-like float (n)
            Gas density at r_sample [cm^-3]
        rScale : float 
            Scaling length for spatial sampling [Mpc]
        eScale : float
            Scaling energy for energy sampling [GeV]
        mode_exp : float
            2 for annihilation, 1 for decay
        delta : float
            Diffusion power-spectrum index
        diff0 : float
            Diffusion constant [cm^2 s^-1]
        lossOnly : boolean
            Flag that sets diffusion on or off
        uPh : float
            Ambient photon energy density [eV cm^-3]
        Delta_ti : float
            Initial time-step [s]
        max_t_part : int
            Number of iterations between time-step reductions
        Delta_t_reduction : float
            Reduction factor for time-step after max_t_part iterations
        Delta_t_min : float
            Smallest time-step in use [s]
        
        Returns
        ---------------------------------
        electrons : array-like float (n,m)
            Equilibrium distribution solution [GeV cm^-3]
        """
        print("="*spacer_length)
        print("ADI environment details")
        print("="*spacer_length)
        
        self.effect = "loss" if lossOnly else "all" 
        
        """ Grid setup and log transforms """
        self.r_bins = len(r_sample)
        self.E_bins = len(E_sample)
        self.r_grid = (r_sample*units.Unit("Mpc")).to("cm").value       #[cm] -> check conversion
        self.E_grid = E_sample          #[GeV] -> check conversion
        self.delta = delta
        self.D0 = diff0
        #variable transformations:  r --> r~ ; E --> E~
        self.r0 = (rScale*units.Unit("Mpc")).to("cm").value    #scale variable [cm]
        self.E0 = eScale        #scale variable [GeV]
        def logr(r):             
            return np.log10(r/self.r0)
        def logE(E):             
            return np.log10(E/self.E0)  
        
        #new log-transformed (and now linspaced) grid
        self.logr_grid = logr(self.r_grid)           #[/]
        self.logE_grid = logE(self.E_grid)           #[/]
        
        """ Diffusion/energy loss functions """
        self.loss_constants = {'ICISRF':6.08e-16 + 0.25e-16*(1+z)**4, 'ICCMB': 0.25e-16*(1+z)**4, 'sync':0.0254e-16, 'coul':6.13e-16, 'brem':4.7e-16}

        rho_sample = (rho_sample*units.Unit("Msun/Mpc^3")*constants.c**2).to("GeV/cm^3").value
        dBdr_sample = (dBdr_sample*units.Unit("1/Mpc")).to("1/cm").value
        self.Q = 1/mode_exp*(np.tensordot(rho_sample,np.ones_like(self.E_grid),axes=0)/mx)**mode_exp*np.tensordot(np.ones_like(rho_sample),Q_sample,axes=0)
        Etens = np.tensordot(np.ones(self.r_bins),self.E_grid,axes=0)
        Btens = np.tensordot(b_sample, np.ones(self.E_bins),axes=0)           
        netens = np.tensordot(ne_sample, np.ones(self.E_bins),axes=0)          
        dBdrtens = np.tensordot(dBdr_sample, np.ones(self.E_bins),axes=0)        

        self.D = self.set_D(Btens,Etens)
        self.dDdr = self.set_dDdr(Btens,dBdrtens,Etens)
        self.b = self.set_b(Btens,netens,Etens,uPh=uPh)
        
        """ Physical scales """
        #virial diffusion velocity ->  used to limit the diffusion function so that it respects the speed of light
        # dLim = (r_sample[-1]*units.Unit("Mpc")).to("cm").value*constants.c.to("cm/s").value 
        # diffVelCondition = self.D > dLim
        # #if there are indices where diffVelCondition is true, limit D and dDdr 
        # if len(self.D[diffVelCondition]) > 0: 
        #     self.D = np.where(diffVelCondition,dLim,self.D)
        #     dDdrLim = self.dDdr[diffVelCondition][0]    #[0] selects first index where D > dLim -> use corresponding dDdr value as the limit for the rest of dDdr
        #     self.dDdr = np.where(diffVelCondition,dDdrLim,self.dDdr)

        #timescales
        self.loss_ts = self.E_grid/self.b
        self.diff_ts = (self.r_grid[1]-self.r_grid[0])**2/self.D  
        loss_min = np.min(self.loss_ts)
        diff_min = np.min(self.diff_ts)

        """ Step sizes """
        self.Delta_r = self.logr_grid[1]-self.logr_grid[0]          #[/]  
        self.Delta_E = self.logE_grid[1]-self.logE_grid[0]          #[/]
        
        if self.benchmark_flag is True:
            self.const_Delta_t = True
            
        if self.const_Delta_t is False:
            #accelerated method
            self.Delta_ti = (Delta_ti*units.Unit('yr')).to('s').value   #large initial timestep to cover all possible timescales
            self.smallest_Delta_t = (Delta_t_min*units.Unit('yr')).to('s').value    #value of Delta_t at which iterations stop when convergence is reached
            self.max_t_part = max_t_part
            self.Delta_t_reduction = Delta_t_reduction
        elif self.const_Delta_t is True:    
            #choose smallest (relevant) timescale as the initial timestep
            if self.effect == "loss":
                self.Delta_ti = loss_min                                #[s] 
            elif self.effect == "diffusion":
                self.Delta_ti = diff_min                                #[s]
            elif self.effect == "all":
                self.Delta_ti = np.min([loss_min,diff_min])             #[s]   
        
        #final value for Delta_ti
        stability_factor = 0.1 if self.benchmark_flag is True else 1.0  #factor that modifies Delta_t by a certain amount (can be used to be 'safely' beneath the timescale of the effects for example)  
        self.Delta_t = self.Delta_ti*stability_factor        #[s]     

        print(f"Included effects: {self.effect}")
        print(f"Domain grid sizes: r_bins: {self.r_bins}, E_bins: {self.E_bins}")
        print(f"Step sizes: Delta_r = {self.Delta_r:.2g}, Delta_E = {self.Delta_E:.2g}")
        print(f"Initial time step: Delta_t = {(self.Delta_t*units.Unit('s')).to('yr').value:.2g} yr")
        print(f"Constant time step: {self.const_Delta_t}\n")
        
        """ADI method """
        print("="*spacer_length)
        print("ADI run details")
        print("="*spacer_length)
        return self.adi_2D(self.Q).transpose()       
        
    """ 
    Function definitions 
    """
    def set_D(self,B,E):
        """
        Diffusion function

        Arguments
        -------------------------
        B : array-like float (n,m)
            Magnetic field strength [uG] 
        E : array-like float (n,m)
            Energy array [GeV]
        
        Returns
        -------------------------
        D : array-like float (n,m)
            Diffusion function result [cm^2 s^-1]
        """
        #set and return diffusion function [cm^2 s^-1]
        D0 = self.D0     #[D0] = cm^2 s^-1
        with np.errstate(divide="ignore",invalid="ignore"):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'overflow')
                #D = D0*d0**(1-alpha)*E**alpha*B**(-alpha)
                D = D0*E**self.delta*(B/np.max(B))**(-self.delta)
        self.D = D
        return D
    
    def set_dDdr(self,B,dBdr,E):
        """
        Diffusion function derivative

        Arguments
        -------------------------
        B : array-like float (n,m)
            Magnetic field strength [uG]
        dBdr : array-like float (n,m)
            Magnetic field derivative [uG cm^-1] 
        E : array-like float (n,m)
            Energy array [GeV]
        
        Returns
        -------------------------
        dDdr : array-like float (n,m)
            Diffusion function derivative [cm s^-1]
        """
        #set and return spatial derivative of diffusion function [cm s^-1]
        D0 = self.D0     #[D0] = cm^2 s^-1

        #prefactor (pf) needed for log-transformed derivative 
        pf = np.tile(self.r_prefactor(np.arange(self.r_bins)),(self.E_bins,1)).transpose()
        with np.errstate(divide="ignore",invalid="ignore"):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'overflow')
                #dDdr = -(1.0/pf*D0*alpha)*d0**(1-alpha)*B**(-alpha-1)*dBdr*E**alpha
                dDdr = -(1.0/pf*D0*self.delta)*(B/np.max(B))**(-self.delta-1)*dBdr/np.max(B)*E**self.delta
        self.dDdr = dDdr
        return dDdr
        
    def set_b(self,B,ne,E,uPh=0.0):
        """
        Energy-loss function

        Arguments
        -------------------------
        B : array-like float (n,m)
            Magnetic field strength [uG]
        ne : array-like density (n,m)
            Gas density [cm^-3] 
        E : array-like float (n,m)
            Energy array [GeV]
        uPh : float
            Photon energy density [eV cm^-3]
        
        Returns
        -------------------------
        b : array-like float (n,m)
            Energy-loss function result [GeV s^-1]
        """
        #set and return energy loss function [GeV s^-1]
        #The final 1/me factor is needed because emissivity integrals are all dimensionless
        #i.e. they are integrated over gamma not E, solution to diffusion equation is prop to 1/b fixing the emissivity dimensions
        b = self.loss_constants
        me = (constants.m_e*constants.c**2).to("GeV").value      #[me] = GeV/c^2 
        eloss = 0.76e-16*uPh*E**2 + b['ICCMB']*E**2 + b['sync']*E**2*B**2 + b['brem']*ne*E
        with np.errstate(divide="ignore",invalid="ignore"):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'overflow')
                coulomb = b['coul']*ne*(1+np.log(E/me/ne)/75.0)
        coulomb = np.where(np.logical_or(np.isnan(coulomb),np.isinf(coulomb)),0.0,coulomb)
        eloss += coulomb
        self.b = eloss
        return eloss
    
    
    """ 
    Prefactors for derivative log transformations 
    """
    def r_prefactor(self,i):
        """
        Normalisation factor for doing log-spaced grid

        Arguments
        -----------------------
        i : int
            Spatial position index
        
        Returns
        -----------------------
        Normalisation factor [cm^-1]
        """
        return (10**self.logr_grid[i]*np.log(10)*self.r0)**-1

    def E_prefactor(self,j):
        """
        Normalisation factor for doing log-spaced grid

        Arguments
        -----------------------
        j : int
            Energy index
        
        Returns
        -----------------------
        Normalisation factor [GeV^-1]
        """
        return (10**self.logE_grid[j]*np.log(10)*self.E0)**-1     
    
    
    """ 
    Alpha coefficient functions 
    
    These support use for both vectorised input (for block matrix solution) or 
    regular integer input (for standard loop solution). 
    Inputs are indices which represent the grid positions of either radius (i) or energy (j)
    """         
    def r_alpha1(self,i,j):
        """
        First spatial propagation coefficient

        Arguments
        -----------------------
        i : int
            Spatial position index
        j : int
            Energy index
        
        Returns
        -----------------------
        Alpha_1 coefficient [cm^-2]
        """
        alpha = np.zeros(i.shape)
        alpha[:] = self.Delta_t*self.r_prefactor(i)**2*(-(np.log10(10)*self.D[i,j] + self.dDdr[i,j])/(2*self.Delta_r) + self.D[i,j]/self.Delta_r**2)
        
        return alpha
            
    def r_alpha2(self,i,j):
        """
        Second spatial propagation coefficient

        Arguments
        -----------------------
        i : int
            Spatial position index
        j : int
            Energy index
        
        Returns
        -----------------------
        Alpha_2 coefficient [cm^-2]
        """
        alpha = np.zeros(i.shape)
        alpha[1:] = self.Delta_t*self.r_prefactor(i[1:])**2*(2*self.D[i[1:],j]/self.Delta_r**2)
        alpha[0] = self.Delta_t*self.r_prefactor(0)**2*4*self.D[0,j]/self.Delta_r**2
        
        return alpha
            
    def r_alpha3(self,i,j):
        """
        Third spatial propagation coefficient

        Arguments
        -----------------------
        i : int
            Spatial position index
        j : int
            Energy index
        
        Returns
        -----------------------
        Alpha_3 coefficient [cm^-2]
        """
        alpha = np.zeros(i.shape)
        alpha[0] = self.Delta_t*self.r_prefactor(0)**2*4*self.D[0,j]/self.Delta_r**2 
        alpha[1:] = self.Delta_t*self.r_prefactor(i[1:])**2*((np.log10(10)*self.D[i[1:],j] + self.dDdr[i[1:],j])/(2*self.Delta_r) + self.D[i[1:],j]/self.Delta_r**2)

        return alpha
         
    def E_alpha1(self,i,j):
        """
        First energy propagation coefficient

        Arguments
        -----------------------
        i : int
            Spatial position index
        j : int
            Energy index
        
        Returns
        -----------------------
        Alpha_1 coefficient [GeV^-1]
        """
        return np.zeros(np.size(j))     
    
    def E_alpha2(self,i,j):
        """
        Second energy propagation coefficient

        Arguments
        -----------------------
        i : int
            Spatial position index
        j : int
            Energy index
        
        Returns
        -----------------------
        Alpha_2 coefficient [GeV^-1]
        """
        return self.Delta_t*self.E_prefactor(j)*self.b[i,j]/self.Delta_E
    
    def E_alpha3(self,i,j):
        """
        Third energy propagation coefficient

        Arguments
        -----------------------
        i : int
            Spatial position index
        j : int
            Energy index
        
        Returns
        -----------------------
        Alpha_3 coefficient [GeV^-1]
        """
        alpha = np.zeros(j.shape)
        alpha[:-1] = self.Delta_t*np.array([self.E_prefactor(j[:-1]+1)*self.b[i,j[:-1]+1]/self.Delta_E]) 
        alpha[-1] = self.Delta_t*np.array([self.E_prefactor(j[-1])*self.b[i,j[-1]]/self.Delta_E])

        return alpha
        
    def spmatrices_loss(self):
        """ 
        A,B Matrix constructors (for sparse block matrices)
        Format is the same in each case - define the matrix diagonals (k_ - u=upper,m=middle,l=lower)
        then construct matrix from those diagonals. 

        Returns
        --------------------------
        Loss matrices [GeV^-1]
        """
        I = self.r_bins
        J = self.E_bins
        IJ = I*J

        #initialise full diagonals for coefficient matrices (need zeros for unassigned indices)
        k_u = np.zeros(IJ)
        k_mA = np.zeros(IJ)
        k_mB = np.zeros(IJ)
        k_l = np.zeros(IJ)    
        
    	#populate diagonals block-by-block. Note upper+lower have J-1 non-zero elements (with 0's at end points)    
        for i in np.arange(I):
            k_u[i*J:(i+1)*J-1] = -self.E_alpha3(i,np.arange(J)[:-1])/2
            k_mA[i*J:(i+1)*J] = 1+self.E_alpha2(i,np.arange(J))/2
            k_mB[i*J:(i+1)*J] = 1-self.E_alpha2(i,np.arange(J))/2
            k_l[i*J:(i+1)*J-1] = -self.E_alpha1(i,np.arange(J)[1:])/2
            
        #A, B matrix constructors from k diagonals              
        loss_A = sparse.diags(diagonals=[k_l,k_mA,k_u],offsets=[-1,0,1],shape=(IJ,IJ),format="csr")
        loss_B = sparse.diags(diagonals=[-k_l,k_mB,-k_u],offsets=[-1,0,1],shape=(IJ,IJ),format="csr")
        
        return (loss_A,loss_B)
    
    def spmatrices_diff(self):
        """ 
        A,B Matrix constructors (for sparse block matrices)
        Format is the same in each case - define the matrix diagonals (k_ - u=upper,m=middle,l=lower)
        then construct matrix from those diagonals. 

        Returns
        --------------------------
        Diffusion matrices [cm^-2]
        """
        I = self.r_bins
        J = self.E_bins
        IJ = I*J
         
        k_u = np.zeros(IJ)
        k_mA = np.zeros(IJ)
        k_mB = np.zeros(IJ)
        k_l = np.zeros(IJ)    
        
        #Note upper+lower diagonals have I-1 non-zero elements (with 0's at end points)
        for j in np.arange(J):
            k_u[j*I:(j+1)*I-1] = -self.r_alpha3(np.arange(I)[:-1],j)/2
            k_mA[j*I:(j+1)*I] = 1+self.r_alpha2(np.arange(I)[:],j)/2
            k_mB[j*I:(j+1)*I] = 1-self.r_alpha2(np.arange(I)[:],j)/2
            k_l[j*I:(j+1)*I-1] = -self.r_alpha1(np.arange(I)[1:],j)/2
                          
        diff_A = sparse.diags(diagonals=[k_l,k_mA,k_u],offsets=[-1,0,1],shape=(IJ,IJ),format="csr")
        diff_B = sparse.diags(diagonals=[-k_l,k_mB,-k_u],offsets=[-1,0,1],shape=(IJ,IJ),format="csr")

        return (diff_A,diff_B)

    def adi_2D(self,Q):
        """
        Solve 2-D diffusion/loss transport equation using the ADI method. 
        
        The general matrix equation being sovled is A*psi1 = B*psi0 + Q,
        where psi == dn/dE is the electron distribution function,
        A, B are coefficient matrices determined from the ADI scheme,
        Q is the electron source function from DM annihilations.
        
        The equation is solved iteratively until convergence is reached. 
        See below for explanation on convergence conditions.
   
        Arguments
        ---------------------------    
        Q : array-like float (n,m)
            DM annihilation source function

        Returns
        ---------------------------
        psi - array-like float (n,m)
            Electron equilibrium function [GeV cm^-3]
        """        
        adi_start = time.perf_counter()
        
        """ Preliminary setup """
        #create A,B matrices for initial timestep        
        if self.effect in {"loss","all"}:
            (loss_A,loss_B) = self.spmatrices_loss()
            print(f"Sparsity of loss matrices A, B: {(np.prod(loss_A.shape)-loss_A.nnz)*100/(np.prod(loss_A.shape)):.3f}%")     
        if self.effect in {"diffusion","all"}:
            (diff_A,diff_B) = self.spmatrices_diff()        
            print(f"Sparsity of diffusion matrices A, B: {(np.prod(diff_A.shape)-diff_A.nnz)*100/(np.prod(diff_A.shape)):.3f}%")
            
        #set initial and boundary conditions
        psi = np.zeros_like(Q)  
        psi[-1,:] = 0.0

        #set convergence and timescale parameters
        convergence_check = False               #main convergence flag to break loop
        stability_check = False                 #flag for stability condition between iterations
        loss_ts_check = False                   #psi_ts > loss_ts check
        diff_ts_check = False                   #psi_ts > diff_ts check
        ts_check = False                        #combination of ts_losscheck and ts_diffcheck
        benchmark_check = False                 #np.all(dpsidt==0), only for benchmarking runs
        rel_diff_check = False                  #relative difference between (t-1) and (t) < stability_tol
        psi_ts = np.empty(psi.shape)            #for calculating psi_ts
        psi_prev = np.empty(psi.shape)          #copy of psi at t-1, for determining stability and convergence checks
        Delta_t_reduction = self.Delta_t_reduction    #factor by which to reduce Delta_t during timestep-switching in accelerated method
        stability_tol = 1.0e-5                  #relative difference tolerance between iterations (for stability_check)
        print(f"Stability tolerance: {stability_tol}")

        #other loop items
        t = 0                                   #total iteration counter 
        t_part = 0                              #iteration counter for each Delta_t 
        t_elapsed = 0                           #total amount of time elapsed during solution (t_part*Delta_t for each Delta_t)       
        max_t = np.int64((np.log(self.Delta_t/self.smallest_Delta_t)/np.log(1/self.Delta_t_reduction)+6)*self.max_t_part)#1e4    
        #maximum total number of iterations (fallback if convergence not reached - roughly 300 iterations per second) 
        max_t_part = self.max_t_part            #maximum number of iterations for each value of Delta_t 
        
        I = self.r_bins
        J = self.E_bins

        #create list of snapshots for animation
        if self.animation_flag is True:
            snapshot = (psi.copy()[:-1],self.Delta_t)
            self.snapshots = [snapshot]

        """ Main ADI loop """
        print("Beginning ADI solution...")
        while not(convergence_check) and (t < max_t):            
            """ 
            Convergence, stability and Delta_t switching
            
            The iterative solution is only stopped if a set of conditions are 
            satisfied, as described below. 
            
            Universal Convergence condition: 
            (c1) - timescale of psi distribution change is greater than energy 
                   loss and/or diffusion timescales (ts_check).
            
            Stability conditions for each timestep value:            
            (s1) - if const_Delta_t is True, the relative difference between 
                   distribution snapshots is less than set tolerance 
            (s2) - if const_Delta_t is False, then the number of iterations 
                   between timestep values (t_part) should be limited by some 
                   predetermined value (t_part_max).
            
            'Benchmark' case:
            (b1) - the timescale of psi distribution -> infinity. The exact 
                   condition being imposed is np.all(dpsidt==0).
                   These runs will always have a constant timestep determined 
                   by the minimum timescale of all the effects. 
    
            'Accelerated' method:
            (a1) - Convergence should only be reached after Delta_t reaches its 
                   smallest value (Delta_t < smallest_Delta_t).
                   If const_Delta_t is False, Delta_t is sequentially reduced 
                   by some predetermined factor (Delta_t_reduction). 
            (a2) - If rel_diff is satisfied together with (a1), ie. at the
                   smallest timestep, override (c1) and allow convergence
            """
            if t>1:
                with np.errstate(divide="ignore",invalid="ignore"):
                    rel_diff = np.abs(psi[:-1]/psi_prev[:-1]-1.0)
                    rel_diff = np.where(np.isnan(rel_diff),0.0,rel_diff)
                    rel_diff_check = bool(np.all(rel_diff < stability_tol))    #[:-1] slice to ignore boundary condition, type conversion because np.bool != bool, get unexpected results sometimes 

                #stability conditions - s1,s2
                if self.const_Delta_t:
                    stability_check = rel_diff_check 
                else:
                    stability_check = t_part > max_t_part
                
                #timescale for psi distribution changes - c1
                dpsidt = (psi[:-1]-psi_prev[:-1])/self.Delta_t
                with np.errstate(divide="ignore",invalid="ignore"): #gets rid of divide by 0 warnings (when psi converges this timescale should tend to inf)
                    psi_ts = np.abs(psi[:-1]/dpsidt)    
                
                #set relevent timescale conditions for each effect
                loss_ts_check = np.all(psi_ts > self.loss_ts[:-1])
                diff_ts_check = np.all(psi_ts > self.diff_ts[:-1])
                if self.effect == "loss":
                    ts_check = loss_ts_check
                elif self.effect == "diffusion":
                    ts_check = diff_ts_check
                elif self.effect == "all":
                    ts_check = loss_ts_check and diff_ts_check 
            
                #diagnostic benchmark check for machine-accuracy dpsidt convergence - b1
                if self.benchmark_flag is True:
                    with np.errstate(divide="ignore"):
                        benchmark_check = np.all(np.abs(dpsidt) == 0)

            #check for convergence if iterations are stable (s1, s2)
            if stability_check:
                if self.benchmark_flag:
                    #benchmark case (b1 + c1)
                    if ts_check and benchmark_check:
                        convergence_check = True
                        break
                else:
                    #non-benchmark cases
                    if self.const_Delta_t:
                        #constant time step (c1)
                        if ts_check: 
                            convergence_check = True
                            break
                    else:
                        #accelerated method 
                        if self.Delta_t > self.smallest_Delta_t:
                            #reduce Delta_t and start again
                            self.Delta_t *= Delta_t_reduction
                            #print(f"Timescale switching activated, Delta t changing to: {(self.Delta_t*units.Unit('s')).to('yr').value:.2g} yr")
                            #print(f"Numer of iterations since previous Delta t: {t_part}\n")
                            t_part = 0
                            
                            #reconstruct A, B matrices with new Delta_t
                            if self.effect in {"loss","all"}:
                                (loss_A,loss_B) = self.spmatrices_loss()
                            if self.effect in {"diffusion","all"}:
                                (diff_A,diff_B) = self.spmatrices_diff()  
                        
                        elif self.Delta_t < self.smallest_Delta_t:
                            if ts_check or rel_diff_check:  
                                #psi has satisfied (a1 + c1) or (a1 + a2) with the lowest timestep - end iterations
                                # print(f"Delta t at lowest value: {(self.Delta_t*units.Unit('s')).to('yr').value:.2g} yr")
                                # print(f"Numer of iterations since previous Delta t: {t_part}\n")
                                convergence_check = True
                                break
                    
            #convergence checks failed - store psi_prev and then update psi
            psi_prev = psi.copy()
            """ 
            Matrix solutions 
            
            When reshaping arrays, the order can be given as 'C' ('c' or row-major) 
            or 'F' ('fortran' or column-major) styles
            """
            if self.effect in {"loss","all"}: 
                rhs = loss_B.dot(psi.flatten('C')) + (Q.flatten('C')*self.Delta_t)
                psi = sparse.linalg.spsolve(loss_A, rhs)   
                psi = np.reshape(psi, (I,J), order='C')
                
            if self.effect in {"diffusion","all"}: 
                rhs = diff_B.dot(psi.flatten('F')) + (Q.flatten('F')*self.Delta_t)
                psi = sparse.linalg.spsolve(diff_A, rhs)                   
                psi = np.reshape(psi, (I,J), order='F')
                
            """ Implement boundary condition and update counters """             
            psi[-1,:] = 0       
            t_elapsed += self.Delta_t
            t += 1
            t_part += 1
            progress(t,max_t,prefix="ADI Progess:")
            
            """ progress feedback during loop """ 
            # progress_check = t%(max_t/10)
            # if progress_check == 0 and t!= 0:
            #     print(f"progress to max {max_t:.2g} iterations...{(t/max_t)*100}%")     

            """ Store solution for animation """
            if self.animation_flag is True:
                snapshot = (psi.copy()[:-1],self.Delta_t)
                self.snapshots.append(snapshot)

            """ Debugging breakpoints """
            # if t%1000 == 0:
            #     print(f"debugging - iteration {t}")
        
        #end while loop
        print()
        self.electrons = psi.copy()        #final equilibrium solution

        print("ADI loop completed.")
        print(f"Convergence: {convergence_check}")
        if not convergence_check:
            warning(f"ADI method did not converge! See diagnostics below as to trustworthiness of the solution\nAverage relative change in psi after last step: {np.sum(rel_diff)/np.size(rel_diff)}\nMaximum relative change in psi after last step: {np.max(rel_diff)}")
            unit_fac = units.Unit("cm").to("Mpc")
            from matplotlib import pyplot as plt
            plt.imshow(np.log10(rel_diff),extent=[np.log10(self.E_grid[0]),np.log10(self.E_grid[-1]),np.log10(self.r_grid[-1]*unit_fac),np.log10(self.r_grid[0]*unit_fac)],aspect="auto")
            cb = plt.colorbar()
            cb.set_label(label=r"$\log_{10}\left(\frac{\psi_n}{\psi_{n-1}}\right)$",fontsize=16)
            plt.ylabel(r"log$_{10}$(r/Mpc)",fontsize=16)
            plt.xlabel(r"log$_{10}$(E/GeV)",fontsize=16)
            plt.title("Relative change in solution at final iteration")
            plt.tight_layout()
            plt.show()
        print(f"Total number of iterations: {t}")
        print(f"Total elapsed time at final iteration: {(t_elapsed*units.Unit('s')).to('yr').value:2g} yr")        

        if self.benchmark_flag is True:
            print("="*spacer_length)
            print("Benchmark Run!") 
            print("="*spacer_length)
            print(f"Benchmark test - all(dpsi/dt == 0): {benchmark_check}")        

        print("ADI solution complete.")
        print(f"Total ADI method run time: {time.perf_counter() - adi_start:.4g} s")
               
        return psi  
