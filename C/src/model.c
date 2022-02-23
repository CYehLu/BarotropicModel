#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../include/utils.h"
#include "../include/helper_spatial.h"
#include "../include/helper_time.h"
#include "../include/type.h"


void _compute_rhs(int ny, int nx, double **psi, double dx, double dy, double beta, double ***rhs, double ***vort_tn) {
    *vort_tn = laplacian(ny, nx, psi, dx, dy);
    double **dpsi_dx = dvar_dx(ny, nx, psi, dx);
    double **jacob = jacobian(ny, nx, psi, *vort_tn, dx, dy);
    
    double **f = allocate_2darray(ny, nx);   // laplacian(rhs) = f
    
    #pragma omp parallel for default(none) shared(f, jacob, beta, dpsi_dx, ny, nx)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) 
            f[j][i] = -jacob[j][i] - beta * dpsi_dx[j][i];
    }
    *rhs = poisson_solver(ny, nx, f, dx, dy);
    
    free_2darray(ny, nx, f);
    free_2darray(ny, nx, dpsi_dx);
    free_2darray(ny, nx, jacob);
}

Fields barotropic(int ny, int nx, double **vort0, double dx, double dy, double dt, int store_dt, int n_steps) {
    double BETA = pow(10., -11.);
    double GAMMA = 0.1;
    int n_store_steps = (n_steps-1)/store_dt + 1;
    
    Fields fields;
    fields.vort = allocate_3darray(n_store_steps, ny, nx);
    fields.psi = allocate_3darray(n_store_steps, ny, nx);
    fields.vort[0] = vort0;
    
    // initial conditions
    fields.psi[0] = poisson_solver(ny, nx, vort0, dx, dy);
    int store_idx = 0;
    
    double **psi_tn = allocate_2darray(ny, nx);     // psi(t_{n})
    double **psi_tnp1 = allocate_2darray(ny, nx);   // psi(t_{n+1})
    double **psi_tnm1 = allocate_2darray(ny, nx);   // psi(t_{n-1})
     
    array2d_copy(ny, nx, psi_tn, fields.psi[0]);
    
    // from time_step = 0 to 1: Euler's method
    printf("Step 1 / %d ...     \r", n_steps);
    
    double **vort_tn = allocate_2darray(ny, nx);
    double **rhs = allocate_2darray(ny, nx);
    _compute_rhs(ny, nx, psi_tn, dx, dy, BETA, &rhs, &vort_tn);
    
    #pragma omp parallel for default(none) shared(psi_tnp1, psi_tn, dt, rhs, ny, nx) 
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++)
            psi_tnp1[j][i] = psi_tn[j][i] + dt * rhs[j][i];
    }
    
    // from time_step = n to n+1: leapfrog + RA filter
    double **tmp;
    
    for (int t = 1; t < n_steps; t++) {
        printf("Step %d / %d ...      \r", t, n_steps);
        
        tmp = psi_tnm1;
        psi_tnm1 = psi_tn;   
        psi_tn = psi_tnp1;
        psi_tnp1 = tmp;
        
        _compute_rhs(ny, nx, psi_tn, dx, dy, BETA, &rhs, &vort_tn);
        
        #pragma omp parallel for default(none) shared(psi_tnm1, psi_tn, psi_tnp1, rhs, dt, GAMMA, ny, nx)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) 
                leapfrog_asselin_filter(&(psi_tnm1[j][i]), &(psi_tn[j][i]), &(psi_tnp1[j][i]), rhs[j][i], dt, GAMMA);
        }
        
        // store data
        if ((t % store_dt) == 0) {
            store_idx++;
            array2d_copy(ny, nx, fields.psi[store_idx], psi_tn);
            array2d_copy(ny, nx, fields.vort[store_idx], vort_tn);
        }
    }
    printf("\n");
    
    // free memory
    free_2darray(ny, nx, psi_tn);
    free_2darray(ny, nx, psi_tnm1);
    free_2darray(ny, nx, psi_tnp1);
    free_2darray(ny, nx, vort_tn);
    free_2darray(ny, nx, rhs);
    
    return fields;
}