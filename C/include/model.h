#include "./type.h"

void _compute_rhs(int, int, double **, double, double, double, double ***, double ***);
Fields barotropic(int ny, int nx, double **vort0, double dx, double dy, double dt, int store_dt, int n_steps);