import numpy as np


###
###
### All of these functions are under the assumption of periodic boundary condition.
###
###


def var_shift(var, ip1=False, im1=False, jp1=False, jm1=False):
    """
    Examples:
        res = var_shift(var, ip1=True) -> var[j,i+1] = res[j,i]
        res = var_shift(var, im1=True) -> var[j,i-1] = res[j,i]
        res = var_shift(var, jp1=True) -> var[j+1,i] = res[j,i]
        res = var_shift(var, jm1=True) -> var[j-1,i] = res[j,i]
        
    Parameters
    ----------
    var: 2d array, shape = (ny, nx)
        The variable to be shifted.
    ip1, im1, jp1, jm1: bool
        `i` stands for the x-axis indice, and `j` stands for y-axis indice.
        `ip1` stands for "i plus 1", and `im1` stands for "i minus 1".
        Note that only one of these arguments can be True.
        
    Return
    ------
    res: 2d array, shape = (ny, nx)
        See examples.
        Note that it assums that the boundary conditions are periodic.
        
    Note
    ----
    `var_shift(var, ip1=True)` is equivalent to `numpy.roll(var, shift=-1, axis=1)`
    I use this function instead of numpy.roll because of the efficiency.
    """
    res = np.empty_like(var)
    
    if ip1:
        res[:,-1] = var[:,0]
        res[:,:-1] = var[:,1:]
    elif im1:
        res[:,0] = var[:,-1]
        res[:,1:] = var[:,:-1]
    elif jp1:
        res[-1,:] = var[0,:]
        res[:-1,:] = var[1:,:]
    elif jm1:
        res[0,:] = var[-1,:]
        res[1:,:] = var[:-1,:]
        
    return res

def dvar_dx(var, dx):
    """
    Calculate d(var)/dx
    """
    var_ip1_j = var_shift(var, ip1=True)
    var_im1_j = var_shift(var, im1=True)
    return (var_ip1_j - var_im1_j) / (2*dx)

def dvar_dy(var, dy):
    """
    Calculate d(var)/dy
    """
    var_i_jp1 = var_shift(var, jp1=True)
    var_i_jm1 = var_shift(var, jm1=True)
    return (var_i_jp1 - var_i_jm1) / (2*dy)

def laplacian(var, dx, dy):
    """
    Calculate Laplacian(var) = d^2(var)/dx^2 + d^2(var)/dy^2
    """
    var_ip1_j = var_shift(var, ip1=True)
    var_im1_j = var_shift(var, im1=True)
    var_i_jp1 = var_shift(var, jp1=True)
    var_i_jm1 = var_shift(var, jm1=True)
    
    d2vardx2 = (var_ip1_j - 2*var + var_im1_j) / (dx**2)
    d2vardy2 = (var_i_jp1 - 2*var + var_i_jm1) / (dy**2)
    return d2vardx2 + d2vardy2

def jacobian(var1, var2, dx, dy):
    """
    Calculate J(var1, var2) = d(var1)/dx * d(var2)/dy - d(var1)/dy * d(var2)/dx.
    
    Use the method provided by Arakawa (1966), page 132, last column in table 1.
    
    Parameters
    ----------
    var1, var2: 2d array, shape = (ny, nx)
        The variables be calculated by Jacobian opertor.
    dx, dy: scalar
        The spatial resolution
        
    Return
    ------
    jacob: 2d array, shape = (ny, nx)
        The Jacobian result.
        
    Reference
    ---------
    Akio Arakawa (1966): "Computational design for long-term numerical integration of 
    the equations of fluid motion: Two-dimensional incompressible flow. Part I"
    Journal of Computational Physics, volume 1, issue 1, pages 119-143
    https://doi.org/10.1016/0021-9991(66)90015-5
    """
    ## simple but less efficient solution
    #jpp = dvar_dx(var1) * dvar_dy(var2) - dvar_dy(var1) * dvar_dx(var2)
    #jpc = dvar_dx( var1*dvar_dy(var2) ) - dvar_dy( var1*dvar_dx(var2) )
    #jcp = dvar_dy( var2*dvar_dx(var1) ) - dvar_dx( var2*dvar_dy(var1) )
    
    var1_ip1_j = var_shift(var1, ip1=True)
    var1_im1_j = var_shift(var1, im1=True)
    var1_i_jp1 = var_shift(var1, jp1=True)
    var1_i_jm1 = var_shift(var1, jm1=True)
    var1_ip1_jp1 = var_shift(var1_ip1_j, jp1=True)
    var1_ip1_jm1 = var_shift(var1_ip1_j, jm1=True)
    var1_im1_jp1 = var_shift(var1_im1_j, jp1=True)
    var1_im1_jm1 = var_shift(var1_im1_j, jm1=True)
    
    var2_ip1_j = var_shift(var2, ip1=True)
    var2_im1_j = var_shift(var2, im1=True)
    var2_i_jp1 = var_shift(var2, jp1=True)
    var2_i_jm1 = var_shift(var2, jm1=True)
    var2_ip1_jp1 = var_shift(var2_ip1_j, jp1=True)
    var2_ip1_jm1 = var_shift(var2_ip1_j, jm1=True)
    var2_im1_jp1 = var_shift(var2_im1_j, jp1=True)
    var2_im1_jm1 = var_shift(var2_im1_j, jm1=True)

    # jpp: J^(++), in Arakawa 1966, eq 36
    jpp = 1/(4*dx*dy) * (
        (var1_ip1_j-var1_im1_j) * (var2_i_jp1-var2_i_jm1)
        - (var1_i_jp1-var1_i_jm1) * (var2_ip1_j-var2_im1_j)
    )
    
    # jpc: J^(+*), eq 37
    jpc = 1/(4*dx*dy) * (
        var1_ip1_j * (var2_ip1_jp1-var2_ip1_jm1)
        - var1_im1_j * (var2_im1_jp1-var2_im1_jm1)
        - var1_i_jp1 * (var2_ip1_jp1-var2_im1_jp1)
        + var1_i_jm1 * (var2_ip1_jm1-var2_im1_jm1)
    )
    
    # jcp: J^(*+), eq 38    
    jcp = 1/(4*dx*dy) * (
        var1_ip1_jp1 * (var2_i_jp1-var2_ip1_j)
        - var1_im1_jm1 * (var2_im1_j-var2_i_jm1)
        - var1_im1_jp1 * (var2_i_jp1-var2_im1_j)
        + var1_ip1_jm1 * (var2_ip1_j-var2_i_jm1)
    )
    
    return 1/3 * (jpp + jpc + jcp)

def poisson_fft(field2d, dx=None, dy=None):
    """
    Solve two dimensional Poisson equation
        laplacian(u) = field2d
    by FFT method with double periodic boundary conditions.
    
    Parameters
    ----------
    field2d: 2d array, shape = (ny, nx)
        The right-hand-side of Poisson equation.
    dx, dy: scalar, optional.
        The spatial resolution. Default is 1.
        
    Return
    ------
    u: 2d array, shape = (ny, nx)
        The solution of Poission equation.
    """
    ## ---- old version: using FFT ----
    ## --------------------------------
    #fhat = np.fft.fft2(field2d)
    #
    #ny, nx = field2d.shape
    #p = np.fft.fftfreq(nx, d=dx)[np.newaxis,:]
    #q = np.fft.fftfreq(ny, d=dy)[:,np.newaxis]
    #factor = (-1.j * 2*np.pi * p) ** 2 + (-1.j * 2*np.pi * q) ** 2 
    #factor[0,0] = -1
    #
    #uhat = fhat / factor
    #uhat[0,0] = 0
    #u = np.fft.ifft2(uhat).real
    #return u
    
    ## ---- new version: using rfft ----
    ## ---------------------------------
    fhat = np.fft.rfft2(field2d)   # the right-most dimension will be truncated
    
    ny, nx = field2d.shape
    #p = np.fft.fftfreq(nx, d=dx)[np.newaxis,:fhat.shape[1]]
    p = np.fft.rfftfreq(nx, d=dx)[np.newaxis,:]
    q = np.fft.fftfreq(ny, d=dy)[:,np.newaxis]
    factor = (-1.j*2*np.pi*p)**2 + (-1.j*2*np.pi*q)**2
    factor[0,0] = -1
    
    uhat = fhat / factor
    uhat[0,0] = 0
    
    u = np.fft.irfft2(uhat, s=field2d.shape)
    return u