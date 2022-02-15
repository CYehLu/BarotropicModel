#include <stdlib.h>
#include <string.h>

void _array2d_copy(int n0, int n1, double **dest, double **src) {
    /*
    Copy 2-d array from `src` to `dest`.
    `n0` and `n1` are the array dimension size.
    */
    for (int j = 0; j < n0; j++) 
        memcpy(dest[j], src[j], n1*sizeof(double));
}

double **allocate_2darray(int n0, int n1) {
    double **ptr = malloc(n0 * sizeof(double *));
    for (int j = 0; j < n0; j++) {
        ptr[j] = malloc(n1 * sizeof(double));
    }
    return ptr;
}

double ***allocate_3darray(int n0, int n1, int n2) {
    double ***arr = malloc(n0 * sizeof(double **));
    for (int k = 0; k < n0; k++) {
        arr[k] = malloc(n1 * sizeof(double *));
        for (int j = 0; j < n1; j++) {
            arr[k][j] = malloc(n2 * sizeof(double));
        }
    }
    return arr;
}

void free_2darray(int n0, int n1, double **var){
    for (int j = 0; j < n0; j++) {
        free(var[j]);
    }
    free(var);
}

void free_3darray(int n0, int n1, int n2, double ***arr) {
    for (int k = 0; k < n0; k++) {
        for (int j = 0; j < n1; j++) {
            free(arr[k][j]);
        }
        free(arr[k]);
    }
    free(arr);
}