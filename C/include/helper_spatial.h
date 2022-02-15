/*
double **allocate_2darray(int, int);
void free_2darray(int, int, double **);
*/

double **dvar_dx(int, int, double **, double);
double **dvar_dy(int, int, double **, double);

double **laplacian(int, int, double **, double, double);
double **jacobian(int, int, double **, double **, double, double);
double **poisson_solver(int, int, double **, double, double);