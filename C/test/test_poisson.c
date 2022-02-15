// gcc test_poisson.c -std=c99 ../helper_spatial.c -I../package/include/ -L../package/lib/ -lfftw3 -lm

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../include/utils.h"
#include "../include/helper_spatial.h"

int main(void)
{
    const double PI = 3.1415926;

    int nx = 40; int ny = 40;
    double dx = 0.1*PI, dy = 0.1*PI;
    
    // allocate and assign value to `F`: F = laplacian(u_true)
    double **F = allocate_2darray(ny, nx);
    
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double x = -2.*PI + i * dx;
            double y = -2.*PI + j * dy;
            F[j][i] = cos(x) + sin(y);
        }
    }
    
    // solve `u`
    double **u = poisson_solver(nx, ny, F, dx, dy);
    
    
    // ----- check the poisson result -----
    
    printf(" ----- u[10:20][10:15] ----- \n");
    for (int j = 10; j < 20; j++) {
        for (int i = 10; i < 15; i++) {
            printf("u[j=%d][i=%d] = %f   ", j, i, u[j][i]);
        }
        printf("\n");
    }
    printf("\n");
    
    printf(" ----- laplacian(u)[10:20][10:15] ----- \n");
    double **lap_u = laplacian(nx, ny, u, dx, dy);
    for (int j = 10; j < 20; j++) {
        for (int i = 10; i < 15; i++) {
            printf("lap(u)[j=%d][i=%d] = %f   ", j, i, lap_u[j][i]);
        }
        printf("\n");
    }
    printf("\n");
    
    printf(" ----- F = laplacian(u_true) ----- \n");
    for (int j = 10; j < 20; j++) {
        for (int i = 10; i < 15; i++) {
            printf("F[j=%d][i=%d] = %f   ", j, i, F[j][i]);
        }
        printf("\n");
    }
    printf("\n");
    
    free_2darray(nx, ny, F);
    free_2darray(nx, ny, u);
    free_2darray(nx, ny, lap_u);
    
    return 0;
}