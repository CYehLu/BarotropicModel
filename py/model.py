import numpy as np
from helper_time import leapfrog_asselin_filter
from helper_spatial import dvar_dx, dvar_dx, laplacian, jacobian, poisson_fft


class BarotropicModel:
    def __init__(self, vort0, dx, dy, dt, store_dt, n_steps):
        """
        Parameters
        ----------
        vort0: 2d array
            Initial vorticity field.
        dx, dy: scalar
            Spatial resolution (unit: m).
        dt: scalar
            The time interval for one time step in the model integration (unit: second).
        store_dt: int
            The model states will be outputted every `store_dt` time steps.
        n_steps: int
            The model will integrate for `total_steps` time steps.
        """
        self.vort0 = vort0
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.store_dt = store_dt
        self.n_steps = n_steps
        
        self.beta = 10 ** -11
        self.gamma = 0.1    # the parameter used in leapfrog_asselin_filter
        
        self.ny, self.nx = vort0.shape
        self.n_store_steps = (n_steps-1) // store_dt + 1
        
        self.vort = np.zeros((self.n_store_steps, self.ny, self.nx))
        self.vort[0,:,:] = vort0
        self.psi = np.zeros_like(self.vort)
        self.u = None
        self.v = None
        
    def _compute_rhs(self, psi, dx, dy, beta):
        """
        Compute the RHS: 
            dpsi_dx = rhs = inverse_laplacian(-Jacobian(psi, laplacian(psi)) - beta * dpsi_dx)
            
        Return
        ------
        (rhs, vort), where vort = laplacian(psi)
        """
        lap = laplacian(psi, dx, dy)
        jacob = jacobian(psi, lap, dx, dy)
        dpsi_dx = dvar_dx(psi, dx)
        rhs = poisson_fft(-jacob - beta*dpsi_dx, dx, dy)
        return rhs, lap
    
    def run(self):
        dx, dy = self.dx, self.dy
        dt = self.dt
        beta = self.beta
        gamma = self.gamma
        
        # initial condition
        self.psi[0,:,:] = poisson_fft(self.vort[0,:,:], dx, dy)
        store_idx = 0
        
        # from i_step = 0 to 1: Euler's method
        print(f'Step 1 / {self.n_steps} ...     ', end='\r')
        psi_tn = self.psi[0,:,:]
        rhs, vort_tn = self._compute_rhs(psi_tn, dx, dy, beta)
        psi_tnp1 = psi_tn + dt * rhs
        
        # from i_step = n to n+1: leapfrog + RA filter
        for i_step in range(1, self.n_steps):
            print(f'Step {i_step+1} / {self.n_steps} ...     ', end='\r')
            
            psi_tnm1 = psi_tn
            psi_tn = psi_tnp1
            rhs, vort_tn = self._compute_rhs(psi_tn, dx, dy, beta)
            psi_tnp1, psi_tn = leapfrog_asselin_filter(psi_tnm1, psi_tn, rhs, dt, gamma)
            
            if i_step % self.store_dt == 0:
                store_idx += 1
                self.psi[store_idx,:,:] = psi_tn
                self.vort[store_idx,:,:] = vort_tn
                
        print('\n[Done]')