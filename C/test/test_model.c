// gcc test_model.c ../model.c ../helper_spatial.c ../helper_time.c -std=c99 -I../package/include/ -L../package/lib/ -lfftw3 -lm

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// for `output_file`
#include <fcntl.h>
#include <sys/io.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../include/utils.h"
#include "../include/model.h"


void output_file(int n0, int n1, double **var, char *filename) {
    int handle = open(filename, O_CREAT|O_WRONLY, S_IRUSR);
    if (handle != -1) {
        for (int j = 0; j < n0; j++)
            write(handle, var[j], n1*sizeof(var[j]));
        close(handle);
    }
    else {
        printf(" *** Error: Failed to open file: %s\n", filename);
    }
}

void output_file3(int n0, int n1, int n2, double ***var, char *filename) {
    int handle = open(filename, O_CREAT|O_WRONLY, S_IRUSR);
    if (handle != -1) {
        for (int t = 0; t < n0; t++) {
            for (int j = 0; j < n1; j++)
                write(handle, var[t][j], n2*sizeof(var[t][j]));
        }
        close(handle);
    }
    else {
        printf(" *** Error: Failed to open file: %s\n", filename);
    }
}

void shift(int n0, int n1, double ***var, int k, int axis) {
    /* 
    Assume: -n0 < k < n0, if axis = 0
            -n1 < k < n1, if axis = 1
    */
    
    double var_shift[n0][n1];   
    
    if (axis == 0) {
        for (int j = 0; j < n0; j++) {
            for (int i = 0; i < n1; i++)
                var_shift[j][i] = (*var)[(j+n0-k)%n0][i];
        }
    }
    else if (axis == 1) {
        for (int j = 0; j < n0; j++) {
            for (int i = 0; i < n1; i++)
                var_shift[j][i] = (*var)[j][(i+n1-k)%n1];
        }
    }
    else {
        printf(" *** axis should be 0 or 1.\n");
        return;
    }
    
    // copy the content from var_shift to var
    for (int j = 0; j < n0; j++) {
        for (int i = 0; i < n1; i++) 
            (*var)[j][i] = var_shift[j][i];
    }
}


int main(void) {    
    // domain configuration
    double dx = 45000., dy = 45000.;
    double xmin = -1800000., xmax = 1800000.;
    double ymin = -1800000., ymax = 1800000.;
    int nx = (int) ((xmax-xmin) / dx + 1);
    int ny = (int) ((ymax-ymin) / dy + 1);
    
    // initial vorticity
    double **vort0 = allocate_2darray(ny, nx);
    
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double x = xmin + i * dx;
            double y = ymin + j * dy;
            vort0[j][i] = 0.0002 * exp(-(x*x+y*y) * 5 * pow(10., -11.));
        }
    }
    
    shift(ny, nx, &vort0, -20, 0);
    shift(ny, nx, &vort0, 20, 1);
    
    // run model
    double dt = 240.;
    int store_dt = 30;
    int n_steps = 2160;
    
    Fields res = barotropic(ny, nx, vort0, dx, dy, dt, store_dt, n_steps);
    int time_size = (n_steps-1)/store_dt + 1;
    
    output_file3(time_size, ny, nx, res.vort, "vort.dat");
    output_file3(time_size, ny, nx, res.psi, "psi.dat");
    
    free_2darray(ny, nx, vort0);
    return 0;
}