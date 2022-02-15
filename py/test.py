import sys
import numpy as np
import matplotlib.pyplot as plt
from helper_spatial import poisson_fft, jacobian, laplacian, dvar_dy, dvar_dx


class TestBase:
    def __init__(self, xmin=-10, xmax=10, dx=0.01, ymin=-10, ymax=10, dy=0.01):
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        self.ymin = ymin
        self.ymax = ymax
        self.dy = dy
        
    def compute_error(self, est, true):
        error = np.sqrt( np.sum((est-true)**2) / np.sum(true**2) )
        maxerror = np.max(np.abs(est-true))
        return error, maxerror
        
    def plot(self, X, Y, Z1, Z2, Z3, titles):
        """
        Plot 3 axes: contourf(X, Y, Z1), contourf(X, Y, Z2), contourf(X, Y, Z3).
        titles = ['title of the first ax', 'title of the second ax', 'title of the third ax']
        The contourf levels of the second and third ax are same.
        """
        fig, axs = plt.subplots(ncols=3, figsize=(16, 4))
        cn = axs[0].contourf(X, Y, Z1)
        axs[0].set_title(titles[0])
        plt.colorbar(cn, ax=axs[0])

        cn1 = axs[1].contourf(X, Y, Z2)
        axs[1].set_title(titles[1])
        plt.colorbar(cn1, ax=axs[1])

        cn2 = axs[2].contourf(X, Y, Z3, levels=cn1.levels)
        axs[2].set_title(titles[2])
        plt.colorbar(cn2, ax=axs[2])

        return fig, axs

    
class TestPoissonSolver(TestBase):      
    def _gen_data(self):
        X, Y = np.meshgrid(
            np.arange(self.xmin, self.xmax, self.dx), 
            np.arange(self.ymin, self.ymax, self.dy)
        )
        
        A = np.exp(-1/15 * ((X-3)**2+Y**2))
        B = np.exp(-1/20 * ((X+3)**2+Y**2))
        
        # laplacian(true_u) = field2d
        true_u = A + B
        field2d = A * (4/225*(X-3)**2 + 4/225*Y**2 - 4/15) + B * (4/400*(X+3)**2 + 4/400*Y**2 - 4/20)
        return X, Y, true_u, field2d
    
    def test(self, solver_func, plot=True):
        X, Y, true_u, field2d = self._gen_data()
        u = solver_func(field2d, self.dx, self.dy)
        
        # rescaling: the mean magnitude is not important
        u = u - u.mean()
        true_u = true_u - true_u.mean()
        
        error, maxerror = self.compute_error(u, true_u)
        print(f" Error     = {error:.5f}")
        print(f" Max Error = {maxerror:.5f}")
        
        if plot:
            titles = ['RHS of Poisson eq', 'true u', 'u solved by FFT solver']
            fig, axs = self.plot(X, Y, field2d, true_u, u, titles)
            plt.show()
            
            
class TestJacobian(TestBase):
    def _gen_data(self):
        X, Y = np.meshgrid(
            np.arange(self.xmin, self.xmax, self.dx), 
            np.arange(self.ymin, self.ymax, self.dy)
        )
        
        v1 = np.exp(-((X-4)**2+Y**2)/10)
        v2 = np.exp(-((X+4)**2+(Y-3)**2)/10)
        true_jacobian = v1*0.2*(X-4)*v2*0.2*(Y-3) - v1*0.2*Y*v2*0.2*(X+4)
        return X, Y, true_jacobian, v1, v2
    
    def test(self, jacobian_func, plot=True):
        X, Y, true_jac, v1, v2 = self._gen_data()
        jac = jacobian_func(v1, v2, self.dx, self.dy)
        error, maxerror = self.compute_error(jac, true_jac)
        print(f" Error     = {error:.5f}")
        print(f" Max Error = {maxerror:.5f}")
        
        if plot:
            titles = ['var1 (contourf) and var2 (black contour)', 'true Jacobian', 'estimated Jacobian']
            fig, axs = self.plot(X, Y, v1, true_jac, jac, titles)
            axs[0].contour(X, Y, v2, colors='k')
            plt.show()
    
            
class TestLaplacian(TestBase):
    def _gen_data(self):
        X, Y = np.meshgrid(
            np.arange(self.xmin, self.xmax, self.dx), 
            np.arange(self.ymin, self.ymax, self.dy)
        )
        
        # laplacian(var) = true_lap
        var = np.exp(-(X**2+Y**2)/10)
        true_lap = (1/25*X**2*var - 1/5*var) + (1/25*Y**2*var - 1/5*var)
        return X, Y, true_lap, var
    
    def test(self, lapacian_func, plot=True):
        X, Y, true_lap, var = self._gen_data()
        lap = lapacian_func(var, self.dx, self.dy)
        
        error, maxerror = self.compute_error(lap, true_lap)
        print(f" Error     = {error:.5f}")
        print(f" Max Error = {maxerror:.5f}")
        
        if plot:
            titles = ['var', 'true laplacian(var)', 'estimated laplacian(var)']
            fig, axs = self.plot(X, Y, var, true_lap, lap, titles)
            plt.show()

            
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('miss argument: {poisson, jacobian, laplacian}')
        print('example:')
        print('    "python test.py poisson laplacian" for testing poisson solver and laplacian operator')
        print('    "python test.py all" for testing all functions')
        sys.exit()
        
    if sys.argv[1] == 'all':
        print(' ----- test poisson -----')
        TestPoissonSolver().test(poisson_fft)
        print()
        
        print(' ----- test jacobian -----')
        TestJacobian().test(jacobian)
        print()

        print(' ----- test laplacian -----')
        TestLaplacian().test(laplacian)
        print()

    else:
        
        for test in sys.argv[1:]:
            if test.lower() == 'poisson':
                print(' ----- test poisson -----')
                TestPoissonSolver().test(poisson_fft)
                print()
                
            elif test.lower() == 'jacobian':
                print(' ----- test jacobian -----')
                TestJacobian().test(jacobian)
                print()

            elif test.lower() == 'laplacian':
                print(' ----- test laplacian -----')
                TestLaplacian().test(laplacian)
                print()
            
            else:
                print(' ------------------------')
                print(f' unavailable type of test: {test}')
                print()
        
        
        
    
    