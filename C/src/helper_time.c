#include <stdlib.h>

void leapfrog_asselin_filter(double *x_nm1, double *x_n, double *x_np1, double rhs, double dt, double gamma) {
    *x_np1 = *x_nm1 + 2. * dt * rhs;
    *x_n = *x_n + gamma * (*x_nm1 - 2.*(*x_n) + *x_np1);
}