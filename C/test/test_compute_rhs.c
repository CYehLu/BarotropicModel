// gcc test_compute_rhs.c ../model.c ../helper_spatial.c ../helper_time.c -std=c99 -I../package/include/ -L../package/lib/ -lfftw3 -lm

#include <stdlib.h>
#include <stdio.h>
#include "../include/utils.h"
#include "../include/model.h"

int main(void) {       
    int ny = 100, nx = 100;
    double dx = 1., dy = 1.;
    double BETA = 1.;
    
    double **vort_tn = allocate_2darray(ny, nx);
    double **rhs = allocate_2darray(ny, nx);
    double **psi = allocate_2darray(ny, nx);
    
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) 
            psi[j][i] = (double) (j*nx + i);
    }
    
    _compute_rhs(ny, nx, psi, dx, dy, BETA, &rhs, &vort_tn);
    
    // ----- check result -----
    printf("rhs:\n");
    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < 5; i++) {
            printf("rhs[j=%d][i=%d] = %f  ", j, i, rhs[j][i]);
        }
        printf("\n");
    }
    printf("\n");
    
    printf("vort_tn:\n");
    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < 5; i++) {
            printf("vort_tn[j=%d][i=%d] = %f  ", j, i, vort_tn[j][i]);
        }
        printf("\n");
    }
    
    free_2darray(ny, nx, vort_tn);
    free_2darray(ny, nx, rhs);
    free_2darray(ny, nx, psi);
    
    return 0;
}