import numpy as np


def leapfrog_asselin_filter(x_nm1, x_n, rhs, dt, gamma=0.1):
    """
    Use leapfrog scheme and Robert & Asselin filtering to calculate the X_{n+1}:
    
    * (forward step)
      X_{n+1} = X_{n-1}_filtered + 2 * dt * F(X_{n})
    * (filtered step)
      X_{n}_filtered = X_{n} + gamma * (X_{n-1}_filtered - 2*X_{n} + X_{n+1})
    
    Parameters
    ----------
    x_nm1: scalar or array
        X_{n-1}
    x_n: scalar or array
        X_{n}
    rhs: scalar or array
        F(X_{n})
    dt: scalar
        Time resolution.
    gamma: scalar
        Filter parameter
        
    Return
    ------
    x_n_filtered, x_np1: scalar or array
        The former is the filtered X_{n}, and the latter is X_{n+1}.
    """
    x_np1 = x_nm1 + 2 * dt * rhs
    x_n_filtered = x_n + gamma * (x_nm1 - 2*x_n + x_np1)
    return x_np1, x_n_filtered