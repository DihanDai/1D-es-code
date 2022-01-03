void jacobi_recurrence(int K_pce, double alpha, double beta, double *a, double *b);
double tuple_product(double *a, double *b, int K_pce, int *n, int nl, int k);
void QsortI(int *n, int low, int high);
int partition(int n[], int low, int high);
void Gaussian_quadrature1D(double *a, double *b, int num_points, \
  double *weights, double *points);
void opoly1d_eval(double *a, double *b, int num_points, double *points, \
  double K_pce, double *eval);
extern void dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);