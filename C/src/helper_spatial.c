// gcc -c helper_spatial.c -std=c99 -fopenmp -I./package/include/ -L./package/lib/ -lfftw3 -lm

#include <stdlib.h>
#include <fftw3.h>
#include "../include/utils.h"

#define PI 3.1415926


double **dvar_dx(int ny, int nx, double **var, double dx) {
    double **dvardx = allocate_2darray(ny, nx);
    
    #pragma omp parallel for default(none) shared(dvardx, var, dx, nx, ny)
    for (int j = 0; j < ny; j++) { 
        for (int i = 0; i < nx; i++) {
            int ip1 = (i+1) % nx;
            int im1 = (i+nx-1) % nx;
            dvardx[j][i] = (var[j][ip1] - var[j][im1]) / (2.*dx);
        }
    }
    
    return dvardx;
}

double **dvar_dy(int ny, int nx, double **var, double dy) {
    double **dvardy = allocate_2darray(ny, nx);
    
    #pragma omp parallel for default(none) shared(dvardy, var, dy, nx, ny)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int jp1 = (j+1) % ny;
            int jm1 = (j+ny-1) % ny;
            dvardy[j][i] = (var[jp1][i] - var[jm1][i]) / (2.*dy);
        }
    }
    
    return dvardy;
}

double **laplacian(int ny, int nx, double **var, double dx, double dy) {
    double **lap = allocate_2darray(ny, nx);
    
    #pragma omp parallel for default(none) shared(lap, ny, nx, var, dx, dy)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int ip1 = (i+1) % nx;
            int im1 = (i+nx-1) % nx;
            int jp1 = (j+1) % ny;
            int jm1 = (j+ny-1) % ny;
            lap[j][i] = (var[j][ip1] - 2*var[j][i] + var[j][im1]) / (dx*dx)  \
                        + (var[jp1][i] - 2*var[j][i] + var[jm1][i]) / (dy*dy);
        }
    }
    
    return lap;
}

double **jacobian(int ny, int nx, double **var1, double **var2, double dx, double dy) {
    double **jacob = allocate_2darray(ny, nx);
    double scale = 1 / (4.*dx*dy);
    
    #pragma omp parallel for default(none) shared(jacob, scale, ny, nx, var1, var2, dx, dy)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int ip1 = (i+1) % nx;
            int im1 = (i+nx-1) % nx;
            int jp1 = (j+1) % ny;
            int jm1 = (j+ny-1) % ny;
            
            // jpp: J^(++), in Arakawa 1966, eq 36
            double jpp = scale * (                                           \
                (var1[j][ip1]-var1[j][im1]) * (var2[jp1][i]-var2[jm1][i])    \
                - (var1[jp1][i]-var1[jm1][i]) * (var2[j][ip1]-var2[j][im1])  \
            );
            
            // jpc: J^(+*), eq 37
            double jpc = scale * (                                           \
                var1[j][ip1] * (var2[jp1][ip1] - var2[jm1][ip1])             \
                - var1[j][im1] * (var2[jp1][im1] - var2[jm1][im1])           \
                - var1[jp1][i] * (var2[jp1][ip1] - var2[jp1][im1])           \
                + var1[jm1][i] * (var2[jm1][ip1] - var2[jm1][im1])           \
            );
            
            // jcp: J^(*+), eq 38    
            double jcp = scale * (                                           \
                var1[jp1][ip1] * (var2[jp1][i] - var2[j][ip1])               \
                - var1[jm1][im1] * (var2[j][im1] - var2[jm1][i])             \
                - var1[jp1][im1] * (var2[jp1][i] - var2[j][im1])             \
                + var1[jm1][ip1] * (var2[j][ip1] - var2[jm1][i])             \
            );
            
            jacob[j][i] = (jpp + jpc + jcp) / 3.;
        }
    }
    
    return jacob;
}

fftw_complex **_fft(int ny, int nx, double **var) {
    int nx_r2c = nx/2 + 1;
    double *in = (double *) fftw_malloc(ny * nx * sizeof(double));
    fftw_complex *out = (fftw_complex *) fftw_malloc(ny * nx_r2c * sizeof(fftw_complex));
    fftw_plan p = fftw_plan_dft_r2c_2d(ny, nx, in, out, FFTW_ESTIMATE);
    
    // convert `var` to `in`, FFTW only allow this kind of multi-dimensional array
    #pragma omp parallel for default(none) shared(in, var, ny, nx)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            in[j*(nx)+i] = var[j][i];
        }
    }
    
    // fft
    fftw_execute(p);
    
    // store the fft result
    fftw_complex **res = malloc(ny * sizeof(fftw_complex *));
    for (int j = 0; j < ny; j++)
        res[j] = malloc(nx_r2c * sizeof(fftw_complex));
    
    #pragma omp parallel for default(none) shared(res, out, ny, nx_r2c)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx_r2c; i++) {
            res[j][i][0] = out[j*(nx_r2c)+i][0];   // real part
            res[j][i][1] = out[j*(nx_r2c)+i][1];   // imaginary part
        }
    }
    
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    
    return res;
}

double **_ifft(int ny, int nx, fftw_complex **varhat) {
    int nx_r2c = nx/2 + 1;
    fftw_complex *in = (fftw_complex *) fftw_malloc(ny*nx_r2c * sizeof(fftw_complex));
    double *out = (double *) fftw_malloc(ny*nx * sizeof(double));
    fftw_plan p = fftw_plan_dft_c2r_2d(ny, nx, in, out, FFTW_ESTIMATE);
    
    // convert `varhat` to `in`, FFTW only allow this kind of multi-dimensional array
    #pragma omp parallel for default(none) shared(in, varhat, ny, nx_r2c)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx_r2c; i++) {
            in[j*nx_r2c+i][0] = varhat[j][i][0];   // real part
            in[j*nx_r2c+i][1] = varhat[j][i][1];   // imaginary part
        }
    }
    
    // ifft
    fftw_execute(p);
    
    // store and rescale the ifft result
    double **res = allocate_2darray(ny, nx);
    
    #pragma omp parallel for default(none) shared(res, out, ny, nx)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            res[j][i] = out[j*nx+i] / (ny*nx);
        }
    }
    
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    
    return res; 
}

double *_fftfreq(int n, double d) {
    double *freq = malloc(n * sizeof(double));
    
    int index = 0;
    for (int i = 0; i < n/2 + n%2; i++) 
        freq[index++] = (double) i / (n*d);
    
    for (int i = n/2; i < n-n%2; i++)
        freq[index++] = (double) (i - n + n%2) / (n*d);
    
    return freq;
}

double *_rfftfreq(int n, double d) {
    double *freq = malloc((n/2+1) * sizeof(double));
    
    for (int index = 0; index < n/2+1; index++)
        freq[index] = (double) index / (n*d);
    
    return freq;
}

double **poisson_solver(int ny, int nx, double **var, double dx, double dy) {
    fftw_complex **varhat = _fft(ny, nx, var);   // shape of `varhat` = (ny, nx/2+1)
    int nx_r2c = nx/2 + 1;
    
    double *p = _rfftfreq(nx, dx);   // x-axis frequencies. size = nx_r2c
    double *q = _fftfreq(ny, dy);    // y-axis frequencies. size = ny
    
    #pragma omp parallel for default(none) shared(varhat, p, q, ny, nx_r2c)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx_r2c; i++) {            
            double factor = ((i==0) && (j==0)) ? -1. : -4*PI*PI*p[i]*p[i] - 4*PI*PI*q[j]*q[j];
            varhat[j][i][0] /= factor;    // real part
            varhat[j][i][1] /= factor;    // imaginary part
        }
    }
    varhat[0][0][0] = 0.;
    varhat[0][0][1] = 0.;
    
    double **u = _ifft(ny, nx, varhat);
    
    // free memory
    free(p); 
    free(q);
    for (int j = 0; j < ny; j++) free(varhat[j]);
    free(varhat);
    
    return u;
}