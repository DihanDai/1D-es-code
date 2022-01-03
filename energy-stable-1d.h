void evol2stage1d_es1(double *bb, double *trip_prod, double dx, \
  int nx, int K_pce, double grav, double *eval_point, \
  int num_point, bool filter_q, double t, double Tlast, double *u, \
  double *dt, FILE *fp);
void rhs1d_cu_es1(double *u, double *bb, double *trip_prod, double dx, \
  int nx, int K_pce, double grav, double *eval_point, \
  int num_point, bool filter_q, double t, double Tlast, \
  double *dt, double *rhs);  
void evol2stage1d_es2(double *bb, double *bE, double *bW, double *trip_prod, \
  double dx, int nx, int K_pce, double grav, double *eval_point, \
  int num_point, bool filter_q, double t, double Tlast, double *u, \
  double *dt, FILE *fp);  
void rhs1d_cu_es2(double *u, double *bb, double *bE, double *bW, double *trip_prod,\
  double dx, int nx, int K_pce, double grav, double *eval_point, \
  int num_point, bool filter_q, double t, double Tlast, \
  double *dt, double *rhs);
  
void entropy_var(int i, int K_pce, double *uu, double *h, double *u, double *trip_prod, double *bb, double grav, double *e_var);
void flux_es_1d_order2(int i, int K_pce, double *uu, double *uuE, double *uuW, \
double *h, double *hE, double *hW, double *trip_prod, double dx, \
double *wtp, double *wtm, double grav, double *f, double *amax); 
void scale_var_order1(int i, int K_pce, double *uu, double *uuE, double *uuW, \
double *h, double *hE, double *hW, double *trip_prod, \
double *wm, double *wp, double *bb, double dx, double *ent, double grav);
void scale_var_order2(int i, int K_pce, double *wm, double *wp, double *wtp, double *wtm);
void flux_es_1d_order1(int i, int K_pce, double *uu, double *h, double *u, double *trip_prod, double *bb, double *e_var, \
double grav, double *f, double *amax);
  void flux_ec_1d(int i, int K_pce, double *uu, double *h, double *trip_prod, \
  double grav, double *f);
void flux_ec_1d_es2(int i, int K_pce, double *uuE, double *uuW, double *hE, double *hW, double *trip_prod, \
  double grav, double *f);
void source_ec_1d(int i, int K_pce, double *h, double *bb, \
  double dx, double grav, double *trip_prod, double *source);
void source_ec_1d_es2(int i, int K_pce, double *hE, double *hW, double *bE, double *bW,\
  double dx, double grav, double *trip_prod, double *source);
void vel_desing1d(double eps, int i, int K_pce, double *h, \
  double *u, double *trip_prod, double *vx); 
void point_recon1d(double *ent, int i, int K_pce, double *ent_E, \
  double *ent_W);
void point_recon1d_2(double *var, int i, int dim, double *varE, \
  double *varW);    
void sqrt_matrix(double *A, int K_pce, double *A_sqrt);
void pce_matrix(double *trip_prod, int K_pce, double *v, double *P_v);
void correct_inv(double eps, int K_pce, double *P_h, double *P_cor);    
double filter_para(double *vec, int len);
void ghostcell1d(int K_pce, int nx, double *u);
double fminmod(double wr, double ux, double wl, double th);
double dot_prod(int len, double *a, double *b);
/* Auxillary array routines. */

void inverse(int n, double *P);
double array_min(double *vec, int start, int end);
double array_max(double *vec, int start, int end);
void array_abs(int start, int end, double *vec);
double array_sum(double *vec, int start, int end);
void fcopy_array(int len, double *A, double *B);
double fminmod2(double a, double b);
void sym_eigen_decom(int dim, double *J, double *W);
void mat_tran(int m, int n, double *a, double *b);
/* LAPACK, BLAS routines. */

extern void dsyev_(char*, char*, int*, double*, int*, double*, double*, int*,\
  int*);
extern void dgemm_(char*,char*,int*,int*,int*,double*,double*,int*,double*,\
  int*,double*,double*,int*);
extern void dsymv_(char*, int*, double*, double*, int*, double*, int*, double*,\
  double*, int*);
extern void dgeev_(char*, char*, int*, double*,int*, double*, double*, double*,\
  int*,double*, int*, double*, int*, int*);
extern void dgetrf_(int*, int*, double*, int*, int*, int*);
extern void dgetri_(int*, double*, int*, int*, double*, int*, int*);
extern void dgecon_(char*, int*, double*, int*, double*, double*, double*, \
  int*, int*);

extern double h_thresh;