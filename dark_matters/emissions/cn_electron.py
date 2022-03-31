import numpy as np
from scipy import sparse
import time
import logging 
logger = logging.getLogger("CN_diffusion")

#global conversion factors
Mpc_to_cm = 1e6*(3.0857e16)*(1e2)
cm_to_Mpc = 1/Mpc_to_cm
yr_to_s = 365.25*24*60*60
s_to_yr = 1/yr_to_s

class eqns_env:
    #Contains calculations on halo properties and the solution of the transport equation
    
    def __init__(self,effect,max_t_part=100,benchmark_flag=False,const_Delta_t=False,animation_flag=False):
        self.effect = effect            #which effects to include in the solution of the transport equation (in set {"loss","diffusion","all"})

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
        self.constants = None   #dictionary of constants for energy loss functions (dict) [GeV s^-1]

        self.Delta_r = None     #step size for space dimension after transform (dimensionless)
        self.Delta_E = None     #step size for energy dimension after transform (dimensionless)
        self.Delta_t = None     #step size for temporal dimension [s]
        self.Delta_ti = None    #initial value for Delta_t

        self.loss_ts = None     #loss timescale
        self.diff_ts = None     #diffusion timescale

        self.benchmark_flag = benchmark_flag      #flag for whether run should use benchmark convergence condition  
        self.const_Delta_t = const_Delta_t  #flag for using a constant step size or not. If False, Delta_t is reduced during method (accelerated method), if True Delta_t remains constant.
        self.smallest_Delta_t = None #smallest value of Delta_t before final convergence (for accelerated timestep switching method)
        self.max_t_part = max_t_part #maximum number of iterations for each value of Delta_t in accelerated method
        
        self.animation_flag = animation_flag      #flag for whether animations take place or not
        self.snapshots = None   #stores snapshots of psi and Delta_t at each iteration for animation
        
    def setup(self,halo,wimp):
        logger.info("=========================\nEquation environment details\n=========================")
        
        """ Grid setup and log transforms """
        self.r_bins = halo.r_bins
        self.E_bins = wimp.E_bins
        self.r_grid = halo.r_arr*(Mpc_to_cm)        #[cm]
        self.E_grid = wimp.E_arr                    #[GeV]
        
        #variable transformations:  r --> r~ ; E --> E~
        self.r0 = (halo.rvir/halo.cvir)*(Mpc_to_cm)      #scale variable [cm]
        self.E0 = 1.0                                    #scale variable [GeV]
        def logr(r):             
            return np.log10(r/self.r0)
        def logE(E):             
            return np.log10(E/self.E0)  
        
        #new log-transformed (and now linspaced) grid
        self.logr_grid = logr(self.r_grid)           #[/]
        self.logE_grid = logE(self.E_grid)           #[/]
        
        """ Diffusion/energy loss functions """
        self.set_Q(wimp)
        self.constants = {'ICISRF':6.08e-16 + 0.25e-16*(1+halo.z)**4, 'ICCMB': 0.25e-16*(1+halo.z)**4, 'sync':0.0254e-16, 'coul':6.13e-16, 'brem':4.7e-16}

        Etens = np.tensordot(np.ones(self.r_bins),self.E_grid,axes=0)
        Btens = np.tensordot(halo.B, np.ones(self.E_bins),axes=0)           
        netens = np.tensordot(halo.ne, np.ones(self.E_bins),axes=0)          
        dBdrtens = np.tensordot(halo.dBdr, np.ones(self.E_bins),axes=0)        

        self.D = self.set_D(Btens,Etens)
        self.dDdr = self.set_dDdr(Btens,dBdrtens,Etens)
        self.b = self.set_b(Btens,netens,Etens)
        self.dbdE = self.set_dbdE(Btens,netens,Etens)
        
        """ Timescales """
        self.loss_ts = self.E_grid/self.b
        self.diff_ts = (self.r_grid[1]-self.r_grid[0])**2/self.D  #check this
        loss_min = np.min(self.loss_ts)
        diff_min = np.min(self.diff_ts)

        """ Step sizes """
        self.Delta_r = self.logr_grid[1]-self.logr_grid[0]          #[/]  
        self.Delta_E = self.logE_grid[1]-self.logE_grid[0]          #[/]
        
        if self.benchmark_flag is True:
            self.const_Delta_t = True
            
        if self.const_Delta_t is False:
            #accelerated method
            self.Delta_ti = 2e9*yr_to_s                             #large initial timestep to cover all possible timescales
            self.smallest_Delta_t = 1e1*yr_to_s                     #value of Delta_t at which iterations stop when convergence is reached
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
        adi_factor = 0.5 if self.effect in {"all"} else 1.0             #factor to account for multiple dimensions in source function updating (ADI method)
        self.Delta_t = self.Delta_ti*adi_factor*stability_factor        #[s]     

        logger.info(f"Included effects: {self.effect}")
        logger.info(f"Solution method: {self.sol_method}")
        logger.info(f"Domain grid sizes: r_bins: {self.r_bins}, E_bins: {self.E_bins}")
        logger.info(f"Step sizes: Delta_r = {self.Delta_r:.2g}, Delta_E = {self.Delta_E:.2g}")
        logger.info(f"Initial time step: Delta_t = {self.Delta_t*s_to_yr:.2g} yr")
        logger.info(f"Constant time step: {self.const_Delta_t}\n")
        
        """ CN method """
        logger.info("=========================\nCN run details\n=========================")
        self.electrons = self.cn_2D(self.Q)        

        logger.info("\n=========================\nResults\n=========================")        
        logger.info(f"Final electron distribution = \n{self.electrons}")

        
    """ 
    Function definitions 
    """
    def set_Q(self, wimp):
        #set WIMP annihilation particle source function; 2D array (r_grid, E_grid); final units [cm^-3 s^-1] 
        Nx = np.tensordot(wimp.wpair_density,np.ones(self.E_bins),axes=0)   #[cm^-6]
        dnde = np.tensordot(np.ones(self.r_bins),wimp.spec,axes=0)          #[GeV^-1]
        cross_section = 1.0e-26                                             #[cm^3 s^-1]
        
        self.Q = Nx*dnde*cross_section   

    def set_D(self,B,E):
        #set and return diffusion function [cm^2 s^-1]
        D0 = 3.1e28     #[D0] = cm^2 s^-1
        d0 = 2.0        #[d0] = kpc
        alpha = 1.0/3.0
        D = D0*(d0)**(1-alpha)*(B)**(-alpha)*(E)**alpha

        self.D = D
        return D
    
    def set_dDdr(self,B,dBdr,E):
        #set and return spatial derivative of diffusion function [cm s^-1]
        D0 = 3.1e28     #[D0] = cm^2 s^-1
        d0 = 2.0        #[d0] = kpc
        alpha = 1.0/3.0

        #prefactor (pf) needed for log-transformed derivative 
        pf = np.tile(self.r_prefactor(np.arange(self.r_bins)),(self.E_bins,1)).transpose()
        # gen_2Dgraph(pf)
        dDdr = -(1.0/pf*D0*alpha)*(d0)**(1-alpha)*(B)**(-alpha-1)*dBdr*(E)**alpha

        self.dDdr = dDdr
        return dDdr
        
    def set_b(self,B,ne,E,ISRF=0):
        #set and return energy loss function [GeV s^-1]
        b = self.constants
        me = 0.511e-3       #[me] = GeV/c^2 
        eloss = b['ICISRF']*(E)**2 + b['sync']*(E)**2*B**2 + b['coul']*ne*(1+np.log(E/(me*ne))/75.0) + b['brem']*ne*E

        self.b = eloss 
        return eloss
    
    def set_dbdE(self,B,ne,E,ISRF=0):
        #set and return energy derivative of energy loss function [s^-1]
        b = self.constants    
        dbdE = 2*b['ICISRF']*(E) + 2*b['sync']*(E)*B**2 + (b['coul']*ne)/(E*75.0) + b['brem']*ne
        
        self.dbdE = dbdE        
        return dbdE
    
    
    """ 
    Prefactors for derivative log transformations 
    """
    def r_prefactor(self,i):
        return (10**self.logr_grid[i]*np.log(10)*self.r0)**-1

    def E_prefactor(self,j):
        return (10**self.logE_grid[j]*np.log(10)*self.E0)**-1     
    
    
    """ 
    Alpha coefficient functions 
    
    These support use for both vectorised input (for block matrix solution) or 
    regular integer input (for standard loop solution). 
    Inputs are indices which represent the grid positions of either radius (i) or energy (j)
    """         
    def r_alpha1(self,i,j):
        #for block matrix solution
        if isinstance(i, np.ndarray):
            alpha = np.zeros(i.shape)
            alpha[:] = self.Delta_t*self.r_prefactor(i)**2*(-(np.log10(10)*self.D[i,j] + self.dDdr[i,j])/(2*self.Delta_r) + self.D[i,j]/self.Delta_r**2)
        
        return alpha
            
    def r_alpha2(self,i,j):
        if isinstance(i,np.ndarray):
            alpha = np.zeros(i.shape)
            alpha[1:] = self.Delta_t*self.r_prefactor(i[1:])**2*(2*self.D[i[1:],j]/self.Delta_r**2)
            alpha[0] = self.Delta_t*self.r_prefactor(0)**2*4*self.D[0,j]/self.Delta_r**2
        
        return alpha
            
    def r_alpha3(self,i,j):
        if isinstance(i, np.ndarray):  
            alpha = np.zeros(i.shape)
            alpha[0] = self.Delta_t*self.r_prefactor(0)**2*4*self.D[0,j]/self.Delta_r**2 
            alpha[1:] = self.Delta_t*self.r_prefactor(i[1:])**2*((np.log10(10)*self.D[i[1:],j] + self.dDdr[i[1:],j])/(2*self.Delta_r) + self.D[i[1:],j]/self.Delta_r**2)

        return alpha
         
    def E_alpha1(self,i,j):
        if isinstance(j, np.ndarray):
            return np.zeros(np.size(j))     
    
    def E_alpha2(self,i,j):
        if isinstance(j, np.ndarray) or isinstance(j,np.int64):
            return self.Delta_t*self.E_prefactor(j)*self.b[i,j]/self.Delta_E
    
    def E_alpha3(self,i,j):
        if isinstance(j,np.ndarray):
            alpha = np.zeros(j.shape)
            alpha[:-1] = self.Delta_t*np.array([self.E_prefactor(j[:-1]+1)*self.b[i,j[:-1]+1]/self.Delta_E]) 
            alpha[-1] = self.Delta_t*np.array([self.E_prefactor(j[-1])*self.b[i,j[-1]]/self.Delta_E])

        return alpha
        
    """ 
    A,B Matrix constructors (for sparse block matrices)
    Format is the same in each case - define the matrix diagonals (k_ - u=upper,m=middle,l=lower)
    then construct matrix from those diagonals. 
    """
    def spmatrices_loss(self):
        I = self.r_bins
        J = self.E_bins
        IJ = I*J

        #initialise full diagonals for coefficient matrices (need zeros for unassigned indices)
        k_u = np.zeros(IJ)
        k_mA = np.zeros(IJ)
        k_mB = np.zeros(IJ)
        k_l = np.zeros(IJ)    
        
    	#populate diagonals block-by-block. Note upper+lower have J-1 non-zero elements (with 0's at end points)    
        for i in range(I):
            k_u[i*J:(i+1)*J-1] = -self.E_alpha3(i,np.arange(J)[:-1])/2
            k_mA[i*J:(i+1)*J] = 1+self.E_alpha2(i,np.arange(J))/2
            k_mB[i*J:(i+1)*J] = 1-self.E_alpha2(i,np.arange(J))/2
            k_l[i*J:(i+1)*J-1] = -self.E_alpha1(i,np.arange(J)[1:])/2
            
        #A, B matrix constructors from k diagonals              
        loss_A = sparse.diags(diagonals=[k_l,k_mA,k_u],offsets=[-1,0,1],shape=(IJ,IJ),format="csr")
        loss_B = sparse.diags(diagonals=[-k_l,k_mB,-k_u],offsets=[-1,0,1],shape=(IJ,IJ),format="csr")
        
        return (loss_A,loss_B)
    
    def spmatrices_diff(self):
        I = self.r_bins
        J = self.E_bins
        IJ = I*J
         
        k_u = np.zeros(IJ)
        k_mA = np.zeros(IJ)
        k_mB = np.zeros(IJ)
        k_l = np.zeros(IJ)    
        
        #Note upper+lower diagonals have I-1 non-zero elements (with 0's at end points)
        for j in range(J):
            k_u[j*I:(j+1)*I-1] = -self.r_alpha3(np.arange(I)[:-1],j)/2
            k_mA[j*I:(j+1)*I] = 1+self.r_alpha2(np.arange(I)[:],j)/2
            k_mB[j*I:(j+1)*I] = 1-self.r_alpha2(np.arange(I)[:],j)/2
            k_l[j*I:(j+1)*I-1] = -self.r_alpha1(np.arange(I)[1:],j)/2
                          
        diff_A = sparse.diags(diagonals=[k_l,k_mA,k_u],offsets=[-1,0,1],shape=(IJ,IJ),format="csr")
        diff_B = sparse.diags(diagonals=[-k_l,k_mB,-k_u],offsets=[-1,0,1],shape=(IJ,IJ),format="csr")

        return (diff_A,diff_B)

    def cn_2D(self,Q):
        """
        Solve 2-D diffusion/loss transport equation using the Crank-Nicholson 
        method. 
        
        The general matrix equation being sovled is A*psi1 = B*psi0 + Q,
        where psi == dn/dE is the electron distribution function,
        A, B are coefficient matrices determined from the CN scheme,
        Q is the electron source function from DM annihilations.
        
        The equation is solved iteratively until convergence is reached. 
        See below for explanation on convergence conditions.
        
        ---------------------------    
        Parameters
        ---------------------------    
        Q (required) - DM annihilation source function - (2D np array)
        
        ---------------------------
        Returns
        ---------------------------
        psi - electron equilibrium function - (2D numpy array)
        """        
        cn_start = time.perf_counter()
        
        """ Preliminary setup """
        #create A,B matrices for initial timestep        
        if self.effect in {"loss","all"}:
            (loss_A,loss_B) = self.spmatrices_loss()
            logger.info(f"Sparsity of loss matrices A, B: {(np.prod(loss_A.shape)-loss_A.nnz)*100/(np.prod(loss_A.shape)):.3f}%")     
        if self.effect in {"diffusion","all"}:
            (diff_A,diff_B) = self.spmatrices_diff()        
            logger.info(f"Sparsity of diffusion matrices A, B: {(np.prod(diff_A.shape)-diff_A.nnz)*100/(np.prod(diff_A.shape)):.3f}%")
            
        #set initial and boundary conditions
        psi = np.array(Q)  
        psi[-1,:] = 0.0

        #set convergence and timescale parameters
        convergence_check = False               #main convergence flag
        stability_check = False                 #flag for stability condition between distribution snapshots
        loss_ts_check = False                   #psi_ts > loss_ts check
        diff_ts_check = False                   #psi_ts > diff_ts check
        ts_check = False                        #combination of ts_losscheck and ts_diffcheck
        benchmark_check = False                 #np.all(dpsidt==0), only for benchmarking runs
        psi_ts = np.empty(psi.shape)            #for calculating psi_ts
        psi_prev = np.empty(psi.shape)          #copy of psi at t-1, for determining stability and convergence checks
        Delta_t_reduction = 0.5                 #factor by which to reduce Delta_t during timestep-switching
        stability_tol = 1.0e-3                  #relative difference (stability) tolerance
        logger.info(f"Stability tolerance: {stability_tol}")

        #other loop items
        t = 0                                   #total iteration counter 
        t_part = 0                              #iteration counter for each Delta_t 
        max_t = 5e5                             #maximum total number of iterations (fallback if convergence not reached - roughly 300 iterations per second) 
        max_t_part = self.max_t_part            #maximum number of iterations for each value of Delta_t 
        t_elapsed = 0                           #total amount of time elapsed during solution (t_part*Delta_t for each Delta_t)       
        
        I = self.r_bins
        J = self.E_bins

        #create list of snapshots for animation
        if self.animation_flag is True:
            snapshot = (psi.copy()[:-1],self.Delta_t)
            self.snapshots = [snapshot]

        """ Main CN loop """
        print("Beginning CN solution... \n")
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
            """
            if t>1:
                #stability conditions - s1,s2
                if self.const_Delta_t:
                    #[:-1] slice to ignore boundary condition, type conversion because np.bool != bool, get unexpected results sometimes
                    stability_check = bool(np.all(np.abs(psi[:-1]/psi_prev[:-1]-1.0) < stability_tol))
                else:
                    stability_check = t_part > max_t_part
                
                #timescale for psi distribution changes - c1
                dpsidt = (psi[:-1]-psi_prev[:-1])/self.Delta_t
                with np.errstate(divide="ignore"): #gets rid of divide by 0 warnings (when psi converges this timescale should tend to inf)
                    psi_ts = np.abs(psi[:-1]/dpsidt)     
                
                #diagnostic benchmark check for machine-accuracy dpsidt convergence - b1
                if self.benchmark_flag is True:
                    with np.errstate(divide="ignore"):
                        benchmark_check = np.all(np.abs(dpsidt) == 0)
                    #stability_check = stability_check and benchmark_check
                
                #set relevent timescale conditions for each effect
                loss_ts_check = np.all(psi_ts > self.loss_ts[:-1])
                diff_ts_check = np.all(psi_ts > self.diff_ts[:-1])
                if self.effect == "loss":
                    ts_check = loss_ts_check
                elif self.effect == "diffusion":
                    ts_check = diff_ts_check
                elif self.effect == "all":
                    ts_check = loss_ts_check and diff_ts_check 
            
            #check for convergence if iterations are stable (s1, s2)
            if stability_check:
                if self.benchmark_flag:
                    #benchmark case
                    if ts_check and benchmark_check:
                        convergence_check = True
                        break
                else:
                    #non-benchmark cases
                    if self.const_Delta_t:
                        #constant time step
                        if ts_check: 
                            convergence_check = True
                            break
                    else:
                        #accelerated method
                        if self.Delta_t > self.smallest_Delta_t:
                            #reduce Delta_t and start again
                            self.Delta_t *= Delta_t_reduction
                            logger.info(f"Timescale switching activated, Delta t changing to: {self.Delta_t*s_to_yr:.2g} yr")
                            logger.info(f"Numer of iterations since previous Delta t: {t_part}\n")
                            t_part = 0
                            
                            #reconstruct A, B matrices with new Delta_t
                            if self.effect in {"loss","all"}:
                                (loss_A,loss_B) = self.spmatrices_loss()
                            if self.effect in {"diffusion","all"}:
                                (diff_A,diff_B) = self.spmatrices_diff()  
                        
                        elif self.Delta_t < self.smallest_Delta_t and ts_check: 
                            #psi has satisfied c1 condition with the lowest timestep - end iterations
                            logger.info(f"Delta t at lowest value: {self.Delta_t*s_to_yr:.2g} yr")
                            logger.info(f"Numer of iterations since previous Delta t: {t_part}\n")
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
                psi = sparse.linalg.spsolve(loss_A,rhs)   
                psi = np.reshape(psi,(I,J),order='C')
                
            if self.effect in {"diffusion","all"}: 
                rhs = diff_B.dot(psi.flatten('F')) + (Q.flatten('F')*self.Delta_t)
                psi = sparse.linalg.spsolve(diff_A,rhs)                   
                psi = np.reshape(psi, (I,J), order='F')
                
            """ Implement boundary condition and update counters """             
            psi[-1,:] = 0       
            t_elapsed += self.Delta_t
            t += 1
            t_part += 1
            
            """ progress feedback during loop """ 
            progress_check = t%(max_t/10)
            if progress_check == 0 and t!= 0:
                print(f"progress to max {max_t:.2g} iterations...{(t/max_t)*100}%")     

            """ Store solution for animation """
            if self.animation_flag is True:
                snapshot = (psi.copy()[:-1],self.Delta_t)
                self.snapshots.append(snapshot)

            # """ 
            # check neumann boundary condition at psi[0,:]
            # r_grid[0] (the centre of the halo) should have psi r-derivative == 0
            # to check this, calculate the finite forward difference of psi between the r = 0 and r = 1 grid positions.
            # """
            # ntol = 1e-3
            # neumann = (psi[1,:]-psi[0,:])/self.Delta_r
            # if np.all(neumann/psi[0,:] < ntol):
            #     print(f"neumann condition working to within {ntol}")
            # else:
            #     print(f"neumann condition not working to within {ntol}")

            #breakpoints for debugging
            if t%5e3 == 0:
                print(f"iteration {t}")                

            #end while
            
        self.electrons = psi.copy()        #final equilibrium solution

        logger.info("CN loop completed.")
        logger.info(f"Convergence: {convergence_check}")
        logger.info(f"Total number of iterations: {t}")
        logger.info(f"Total elapsed time at final iteration: {t_elapsed*s_to_yr:2g} yr")        

        if self.benchmark_flag is True:
            logger.info("\n=========================\nBenchmark Run!\n=========================") 
            logger.info(f"Benchmark test - all(dpsi/dt == 0): {benchmark_check}")        

        print("CN solution complete.")
        print(f"Total CN method run time: {time.perf_counter() - cn_start:.4g} s")
               
        return psi   
