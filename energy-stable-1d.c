/*
 * Central upwind scheme routines for 1D SGSWE.
 * (Last updated at 09-24-2020)
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>


#include "opoly1d.h" // For orthonormal polynomials routines 

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

double h_thresh = 1e-14;

/*
 * evolo2stage1d_es2 : 1-time-step evolution using two-stage SSPRK for 2nd-order energy-stable scheme.
 *
 * Parameters:
 * [In] bb : double array
 *          cell average of the piecewise linear reconstructed bottom topography
 * [In] bE : double array
 *          east-side reconstructed values of the piecewise linear reconstructed topography
 * [In] bW : double array
 *          west-side reconstructed values of the piecewise linear reconstructed topography
 * [In] trip_prod : double array
 *          array of triple (inner) product of the orthonormal polynomials
 * [In] dx : double
 *          (uniform) spatial grid size
 * [In] nx : int
 *          number of spatial cells
 * [In] K_pce : int
 *          number of expansion terms in PCE.
 * [In] grav : double 
 *          graviational constant
 * [In] eval_point : double array
 *          valuation of the orthonormal polynomials at prescribed quadrature points.
 * [In] num_point : int
 *          number of quadrature point
 * [In] filter_q : boolean
 *          indicator of whether the discharge is filtered.
 * [In] Tlast : double 
 *          the final time.
 * [In] t : double
 *          current time
 * [In/Out] u : double array
 *          On input, u is the current state. On output, u is the state after 1-time-step evolution
 * [Out] dt : double
 *          time step
 * [In/Out] fp : FILE*
 *          file for keeping the energy during iteration.
*/
void evol2stage1d_es2(double *bb, double *bE, double *bW, double *trip_prod, \
  double dx, int nx, int K_pce, double grav, double *eval_point, \
  int num_point, bool filter_q, double t, double Tlast, double *u, \
  double *dt, FILE *fp)
{
  double *rhs, *rhs1, *u1, *htemp, energy;
  double *v1, *v2, *v3, *v4, *h, *uu;
	double dt1, kappa=0.9;
	int i, j, k, iter1 = 0;

  rhs = malloc((nx + 2) * 2 * K_pce * sizeof(double)); // right-hand-side of the evolution at the first stage
  rhs1 = malloc((nx + 2) * 2 * K_pce * sizeof(double)); // right-hand-side of the evolution at the second stage
  u1 = malloc((nx + 2) * 2 * K_pce * sizeof(double)); // intermediate state
  htemp = malloc((num_point) * sizeof(double)); 
  v1 = malloc((K_pce) * sizeof(double)); 
  v2 = malloc((K_pce) * sizeof(double)); 
  v3 = malloc((K_pce) * sizeof(double)); 
  v4 = malloc((K_pce) * sizeof(double)); 
  uu = malloc((nx + 2) * (K_pce) * sizeof(double)); 
  h = malloc((nx + 2) * (K_pce) * sizeof(double)); 

  ghostcell1d(K_pce, nx, u); // ghost cell (outflow boundary for now 12/19/2021)
  rhs1d_cu_es2(u, bb, bE, bW, trip_prod, dx, nx, K_pce, grav, eval_point, \
    num_point, filter_q, t, Tlast, dt, rhs); // compute the right-hand-side of the semi-discrete form at current state.

  // Stage 1:
  while(iter1 < 20)
  {    
    iter1++;
    for(i = 1; i <= nx; i++)
    {
      for(j = 0; j < 2 * K_pce; j++)
      {
        u1[j + i * 2 * K_pce] = \
          (u[j + i * 2 * K_pce] + (*dt) * rhs[j + i * 2 * K_pce]); // compute the intermediate state         
      }
    }
    ghostcell1d(K_pce, nx, u1);
    rhs1d_cu_es2(u1, bb, bE, bW, trip_prod, dx, nx, K_pce, grav, eval_point, \
      num_point, filter_q, t, Tlast, &dt1, rhs1);  // compute the right-hand-side using the intermediate state
  
    if(dt1 < (*dt))
    {
      (*dt) = kappa * dt1; // recompute the intermediate state with a smaller time step if the time step of the second stage exceeds the one in the first stage.
    }
    else
    {
      break;
    }
  }

  // Stage 2:
  for(i = 1; i <= nx; i++)
  {
    for(j = 0; j < 2 * K_pce; j++)
    {
      u[j + i * 2 * K_pce] = 0.5 * (u[j + i * 2 * K_pce] + 
        u1[j + i * 2 * K_pce] + (*dt) * rhs1[j + i * 2 * K_pce]);          
    }
    for(j = 0; j <= K_pce; j++)
    {
      h[j + i * K_pce] = u[j + i * 2 * K_pce] - bb[j + i * K_pce]; // PCE for the water height
    }    
  }
 
  ghostcell1d(K_pce, nx, u);
  for(i = 1; i <= nx; i++)
  {
    vel_desing1d(0, i, K_pce, h, u, trip_prod, uu); // compute the velocity
  }  
  energy = 0;
  for(i = 1; i <= nx; i++)
  {
    for(j = 0; j < K_pce; j++)
    {
      v1[j] = h[j + i * K_pce];
      v2[j] = uu[j + i * K_pce];
      v3[j] = bb[j + i * K_pce];
      v4[j] = u[(j + K_pce) + i * 2 * K_pce];
    }
    energy += dx * (0.5 * (dot_prod(K_pce, v2, v4) + grav * dot_prod(K_pce, v1, v1))+
      grav * dot_prod(K_pce, v1, v3)); // compute the discrete energy
  }    

  // Asserting the water height is positive at quadrature points. 

  for(i = 1; i <= nx; i++)
  {
    for(k = 0; k < num_point; k++)
    {
      for(j = 0; j < K_pce; j++)
      {
        htemp[j] = (u[j + i * 2 * K_pce] - bb[j + i * K_pce]) * eval_point[k + j * num_point];
      }
      assert(array_sum(htemp, 0, K_pce) >= -1e-15);
    }
  }
  fprintf(fp,"%d\t%.16f\t%.16f\n", iter1, *dt, energy);
  free(rhs); free(rhs1); 
  free(u1); free(htemp);
  free(v1); free(v2); free(v3); free(v4);
  free(h); free(uu);
}

/*
 * rhs1d_cu_es2 : right-hand-side for the second-order energy-stable scheme.
 *
 * Parameters:
 * [In] u : double array
 *          u is the state used for constructing the right-hand-side of the semi-discrete form.
 * [In] bb : double array
 *          cell average of the piecewise linear reconstructed bottom topography
 * [In] bE : double array
 *          east-side reconstructed values of the piecewise linear reconstructed topography
 * [In] bW : double array
 *          west-side reconstructed values of the piecewise linear reconstructed topography
 * [In] trip_prod : double array
 *          array of triple (inner) product of the orthonormal polynomials
 * [In] dx : double
 *          (uniform) spatial grid size
 * [In] nx : int
 *          number of spatial cells
 * [In] K_pce : int
 *          number of expansion terms in PCE.
 * [In] grav : double 
 *          graviational constant
 * [In] eval_point : double array
 *          valuation of the orthonormal polynomials at prescribed quadrature points.
 * [In] num_point : int
 *          number of quadrature point
 * [In] filter_q : boolean
 *          indicator of whether the discharge is filtered.
 * [In] t : double
 *          current time
 * [In] Tlast : double 
 *          the final time.
 * [Out] rhs : double array
 *          rhs is the right-hand-side constructed using the input state u.
 * [Out] dt : double
 *          time step
*/
void rhs1d_cu_es2(double *u, double *bb, double *bE, double *bW, double *trip_prod,\
  double dx, int nx, int K_pce, double grav, double *eval_point, \
  int num_point, bool filter_q, double t, double Tlast, \
  double *dt, double *rhs)
{

  int i, j, k;
  double *h, *uu, *htemp, *rhs_temp, cfl;
  double *Hx, *source, eps = pow(dx, 4), *amax, am, *e_var;
  double *uE, *uW, *hE, *hW, *uuE, *uuW, *wtjump;
  double *wm, *wp, *wtm, *wtp;
	//clock_t start, end;
	//double t_filter, t_speeds, t_veld, t_flux1, t_globalt;

  // Memory allocation 
  uu = malloc((nx + 2) * (K_pce) * sizeof(double)); // PCE vector for cell average of the velocity.
  h = malloc((nx + 2) * (K_pce) * sizeof(double)); // PCE vector for cell average of the water height.
  rhs_temp = malloc((K_pce) * sizeof(double)); 
  htemp = malloc((K_pce) * sizeof(double)); 
  Hx = malloc((nx + 2) * (K_pce * 2) * sizeof(double)); // numerical flux
  e_var = malloc((nx + 2) * (K_pce * 2) * sizeof(double)); // PCE vector for the cell average of the entropy variable 
  hE = malloc((nx + 2) * (K_pce) * sizeof(double)); // east-side reconstruction of PCE vector for the water height
  hW = malloc((nx + 2) * (K_pce) * sizeof(double)); // west-side reconstruction of PCE vector for the water height 
  uuE = malloc((nx + 2) * (K_pce) * sizeof(double)); // east-side reconstruction of PCE vector for the velocity 
  uuW = malloc((nx + 2) * (K_pce) * sizeof(double)); // west-side reconstruction of PCE vector for the velocity 
  uE = malloc((nx + 2) * (K_pce * 2) * sizeof(double)); // east-side reconstruction of PCE vector for the state.
  uW = malloc((nx + 2) * (K_pce * 2) * sizeof(double)); // west-side reconstruction of PCE vector for the state.      
  wm = malloc((nx + 2) * (K_pce * 2) * sizeof(double)); // west-side first-order scaled variable
  wp = malloc((nx + 2) * (K_pce * 2) * sizeof(double)); // east-side first-order scaled variable
  wtm = malloc((nx + 2) * (K_pce * 2) * sizeof(double)); // west-side second-order scaled variable
  wtp = malloc((nx + 2) * (K_pce * 2) * sizeof(double)); // east-side first-order scaled variable
  source = malloc((nx + 2) * (K_pce * 2) * sizeof(double)); // energy-conservative source.
  amax = malloc((nx + 2) * sizeof(double));  // maximum local propagation speeds at each cell interface.
  wtjump = malloc((nx + 2) * sizeof(double)); // east-side first-order scaled variable
  // compute the average water height.
  for(i = 1; i <= nx; i++)
  {
    for(j = 0; j <= K_pce; j++)
    {
      h[j + i * K_pce] = u[j + i * 2 * K_pce] - bb[j + i * K_pce];
    }
  }

  // compute the corresponding PCE vector for the velocity (velocity desingularization is applied here).
  for(i = 1; i <= nx; i++)
  {
    vel_desing1d(eps, i, K_pce, h, u, trip_prod, uu);
  }
  
  // compute the corresponding entropy variable.
  for(i = 1; i <= nx; i++)
  {
    entropy_var(i, K_pce, uu, h, u, trip_prod, bb, grav, e_var);   
    //printf("%d %.16f %.16f\n", i, e_var[i*2*K_pce], e_var[K_pce+i*2*K_pce]);
  }  

  // compute the pointwise reconstructed values for the water surface and the velocity. 
  for(i = 1; i <= nx; i++)
  {
    point_recon1d(u, i, 2 * K_pce, uE, uW);
    point_recon1d(uu, i, K_pce, uuE, uuW);
  }

  // compute the pointwise reconstructed values for the water height
  for(i = 1; i <= nx; i++)
  {
    for(j = 0; j <= K_pce; j++)
    {
      hE[j + i * K_pce] = uE[j + i * 2 * K_pce] - bE[j + i * K_pce];
      hW[j + i * K_pce] = uW[j + i * 2 * K_pce] - bW[j + i * K_pce];
    }
  }  

  for(j = 0; j < K_pce; j++)
  {
    // velocity
    uu[j] = uu[j + K_pce];
    uu[j + (nx + 1) * K_pce] = uu[j + (nx) * K_pce];

    // water height
    h[j] = h[j + K_pce];
    h[j + (nx + 1) * K_pce] = h[j + (nx) * K_pce];

    // state variables
    u[j] = u[j + 2 * K_pce];
    u[j + (nx + 1) * 2 * K_pce] = u[j + nx * 2 * K_pce];        
    u[j + K_pce] = u[j + K_pce + 2 * K_pce];
    u[j + K_pce + (nx + 1) * 2 * K_pce] = u[j + K_pce + (nx) * 2 * K_pce];   

    // entropy variable
    e_var[j] = e_var[j + 2 * K_pce];
    e_var[j + (nx + 1) * 2 * K_pce] = e_var[j + nx * 2 * K_pce];                
    e_var[j + K_pce] = e_var[j + K_pce + 2 * K_pce];
    e_var[j + K_pce + (nx + 1) * 2 * K_pce] = e_var[j + K_pce + (nx) * 2 * K_pce];   

    // reconstructed states
    uE[j] = uW[j + 2 * K_pce]; 
    uE[(j + K_pce)] = uW[(j + K_pce) + 2 * K_pce];          
    uW[j + (nx + 1) * 2 * K_pce] = uE[j + 2 * nx * K_pce];
    uW[(j + K_pce) + (nx + 1) * 2 * K_pce] = uE[(j + K_pce) + 2 * nx * K_pce];       
    
    // reconstructed water height
    hE[j] = hW[j + K_pce]; 
    hW[j + (nx + 1) * K_pce] = hE[j + nx * K_pce];

    // reconstructed velocity
    uuE[j] = uuW[j + K_pce]; 
    uuW[j + (nx + 1) * K_pce] = uuE[j + nx * K_pce];
  }

  for(i = 0; i <= nx; i++)
  {
    scale_var_order1(i, K_pce, uu, uuE, uuW, h, hE, hW, trip_prod, \
      wm, wp, bb, dx, e_var, grav);
  }
  for(i = 1; i <= nx; i++)
  {
    scale_var_order2(i, K_pce, wm, wp, wtp, wtm); 
  }

  // Outflow boundary for wtm and wtp
  for(j = 0; j < K_pce; j++)
  {
    // reconstructed scaled entropy variables.
    wtp[j] = wtm[j + 2 * K_pce]; 
    wtp[(j + K_pce)] = wtm[(j + K_pce) + 2 * K_pce];          
    wtm[j + (nx + 1) * 2 * K_pce] = wtp[j + 2 * nx * K_pce];
    wtm[(j + K_pce) + (nx + 1) * 2 * K_pce] = wtp[(j + K_pce) + 2 * nx * K_pce];       
  }
  for(i = 0; i <= nx; i++)
  {
    wtjump[i] = -wtp[K_pce+i * 2 * K_pce]+wtm[K_pce+(i+1)*2*K_pce];
  }
  printf("%.16f\n", array_max(wtjump, 0, nx+1));
  // compute the energy conservative source
  for(i = 0; i < nx+2; i++)
  {
    source_ec_1d(i, K_pce, h, bb, dx, grav, trip_prod, source);
  }

  // compute the second-order numerical flux.
  for(i = 0; i < nx+1; i++)
  {
    flux_es_1d_order2(i, K_pce, uu, uuE, uuW, h, hE, hW, trip_prod, \
      dx, wtp, wtm, grav, Hx, amax);
  }

  // construct the right-hand-side and compute the hyperbolicity-preserving time step.
  cfl = 0.0;
  for(i = 1; i <= nx; i++)
  {
    for(j = 0; j < 2 * K_pce; j++)
    {
      rhs[j + i * 2 * K_pce] = -(Hx[j + 2 * i * K_pce] - Hx[j + 2 * (i - 1) * K_pce])/dx + source[j + 2 * i * K_pce];
    }
    //printf("%d %.16f %.16f\n", i, rhs[i * 2 * K_pce], rhs[K_pce + i * 2 * K_pce]);
    for(k = 0; k < num_point; k++)
    {
      for(j = 0; j < K_pce; j++)
      {
        htemp[j] = h[j + i * K_pce] * \
          eval_point[k + j * num_point];
          
        rhs_temp[j] = rhs[j + i * 2 * K_pce] * \
          eval_point[k + j * num_point];
      }      
      if(array_sum(htemp, 0, K_pce) <= -h_thresh)
      {
        printf("cell %d\n", i);
        for(j = 0; j < K_pce; j++)
        {
          printf("%.16f %.16f\n", h[j + i * K_pce], eval_point[k + j * num_point]);
        }
      }
      assert(array_sum(htemp, 0, K_pce) > -h_thresh);
      
      if(array_sum(rhs_temp, 0, K_pce) < -h_thresh)
      {
        cfl = fmax(cfl, -array_sum(rhs_temp, 0, K_pce)/\
          array_sum(htemp, 0 , K_pce));       
      }
    }     
  }
  // incorporate the hyperbolicity-preserving time step and wave-speed CFL time step
  am = array_max(amax, 1, nx + 1);
  (*dt) = 0.5 * dx / am;
  (*dt) = 0.9 * fmin((*dt), 1.0/cfl);  
  
  if(t + (*dt) > Tlast)
  {
    (*dt) = Tlast - t;
  }
  
  // Free space
  free((void *)wtjump);
  free((void *)wm); free((void *)wp);
  free((void *)wtm); free((void *)wtp);
  free((void *)amax); free((void*)e_var);
  free((void *)hE); free((void*)hW); 
  free((void *)uuE); free((void*)uuW); 
  free((void *)uE); free((void*)uW);       
  free((void *)h); free((void *)uu); 
  free((void *)Hx); free((void *)source);
  free((void *)rhs_temp); free((void *)htemp);
}


/*
 * evolo2stage1d_es1 : 1-time-step evolution using two-stage SSPRK for 1st-order energy-stable scheme.
 *
 * Parameters:
 * [In] bb : double array
 *          cell average of the piecewise linear reconstructed bottom topography
 * [In] trip_prod : double array
 *          array of triple (inner) product of the orthonormal polynomials
 * [In] dx : double
 *          (uniform) spatial grid size
 * [In] nx : int
 *          number of spatial cells
 * [In] K_pce : int
 *          number of expansion terms in PCE.
 * [In] grav : double 
 *          graviational constant
 * [In] eval_point : double array
 *          valuation of the orthonormal polynomials at prescribed quadrature points.
 * [In] num_point : int
 *          number of quadrature point
 * [In] filter_q : boolean
 *          indicator of whether the discharge is filtered.
 * [In] Tlast : double 
 *          the final time.
 * [In] t : double
 *          current time
 * [In/Out] u : double array
 *          On input, u is the current state. On output, u is the state after 1-time-step evolution
 * [Out] dt : double
 *          time step
 * [In/Out] fp : FILE*
 *          file for keeping the energy during iteration.
*/
void evol2stage1d_es1(double *bb, double *trip_prod, double dx, \
  int nx, int K_pce, double grav, double *eval_point, \
  int num_point, bool filter_q, double t, double Tlast, double *u, \
  double *dt, FILE *fp)
{
  double *rhs, *rhs1, *u1, *htemp, *proj_coef, energy;
  double *v1, *v2, *v3, *v4, *h, *uu;
	double dt1, kappa=0.9;
	int i, j, k, iter1 = 0;

  rhs = malloc((nx + 2) * 2 * K_pce * sizeof(double)); // right-hand-side of the evolution at the first stage
  rhs1 = malloc((nx + 2) * 2 * K_pce * sizeof(double)); // right-hand-side of the evolution at the second stage
  u1 = malloc((nx + 2) * 2 * K_pce * sizeof(double)); // intermediate state
  htemp = malloc(K_pce * sizeof(double));
  v1 = malloc((K_pce) * sizeof(double)); 
  v2 = malloc((K_pce) * sizeof(double)); 
  v3 = malloc((K_pce) * sizeof(double)); 
  v4 = malloc((K_pce) * sizeof(double)); 
  uu = malloc((nx + 2) * (K_pce) * sizeof(double)); 
  h = malloc((nx + 2) * (K_pce) * sizeof(double)); 

  ghostcell1d(K_pce, nx, u); // ghost cell (outflow boundary for now 12/19/2021)
  rhs1d_cu_es1(u, bb, trip_prod, dx, nx, K_pce, grav, eval_point, \
    num_point, filter_q, t, Tlast, dt, rhs);  // compute the right-hand-side of the semi-discrete form at current state.

  // Stage 1:
  while(iter1 < 20)
  {    
    iter1++;
    for(i = 1; i <= nx; i++)
    {
      for(j = 0; j < 2 * K_pce; j++)
      {
        u1[j + i * 2 * K_pce] = \
          (u[j + i * 2 * K_pce] + (*dt) * rhs[j + i * 2 * K_pce]);          
      }
    }
    ghostcell1d(K_pce, nx, u1);
    rhs1d_cu_es1(u1, bb, trip_prod, dx, nx, K_pce, grav, eval_point, \
      num_point, filter_q, t, Tlast, &dt1, rhs1);  // compute the right-hand-side using the intermediate state
    if(dt1 < (*dt))
    {
      (*dt) = kappa * dt1; // recompute the intermediate state with a smaller time step if the time step of the second stage exceeds the one in the first stage.
    }
    else
    {
      break;
    }
  }

  // Stage 2:  
  for(i = 1; i <= nx; i++)
  {
    for(j = 0; j < 2 * K_pce; j++)
    {
      u[j + i * 2 * K_pce] = 0.5 * (u[j + i * 2 * K_pce] + 
        u1[j + i * 2 * K_pce] + (*dt) * rhs1[j + i * 2 * K_pce]);          
    }
    for(j = 0; j <= K_pce; j++)
    {
      h[j + i * K_pce] = u[j + i * 2 * K_pce] - bb[j + i * K_pce]; // PCE for the water height
    }    
  }

  ghostcell1d(K_pce, nx, u);
  for(i = 1; i <= nx; i++)
  {
    vel_desing1d(0, i, K_pce, h, u, trip_prod, uu); // compute the velocity
  }  
  energy = 0;
  for(i = 1; i <= nx; i++)
  {
    for(j = 0; j < K_pce; j++)
    {
      v1[j] = h[j + i * K_pce];
      v2[j] = uu[j + i * K_pce];
      v3[j] = bb[j + i * K_pce];
      v4[j] = u[(j + K_pce) + i * 2 * K_pce];
    }
    energy += dx * (0.5 * (dot_prod(K_pce, v2, v4) + grav * dot_prod(K_pce, v1, v1))+
      grav * dot_prod(K_pce, v1, v3));
  }    
  // Asserting the water height is positive. 

  for(i = 1; i <= nx; i++)
  {
    for(k = 0; k < num_point; k++)
    {
      for(j = 0; j < K_pce; j++)
      {
        htemp[j] = (u[j + i * 2 * K_pce] - bb[j + i * K_pce])* eval_point[k + j * num_point];
      }
      if(array_sum(htemp, 0, K_pce) < -1e-15)
      {
        printf("%d %d\n", i, k);
        for(j = 0; j < K_pce; j++)
        {
          printf("%.16f %.16f %.16f\n",u[j + i * 2 * K_pce], bb[j + i * K_pce], eval_point[k + j * num_point]);
        }
      }
      assert(array_sum(htemp, 0, K_pce) >= -1e-15);
    }
  }
  fprintf(fp,"%d\t%.16f\t%.16f\n",iter1, t, energy);
  free(rhs); free(rhs1);
  free(u1); free(htemp);
  free(v1); free(v2); free(v3); free(v4);
  free(h); free(uu);
}

/*
 * rhs1d_cu_es1 : right-hand-side for the first-order energy-stable scheme.
 *
 * Parameters:
 * [In] u : double array
 *          u is the state used for constructing the right-hand-side of the semi-discrete form.
 * [In] bb : double array
 *          cell average of the piecewise linear reconstructed bottom topography
 * [In] trip_prod : double array
 *          array of triple (inner) product of the orthonormal polynomials
 * [In] dx : double
 *          (uniform) spatial grid size
 * [In] nx : int
 *          number of spatial cells
 * [In] K_pce : int
 *          number of expansion terms in PCE.
 * [In] grav : double 
 *          gravitational constant
 * [In] eval_point : double array
 *          valuation of the orthonormal polynomials at prescribed quadrature points.
 * [In] num_point : int
 *          number of quadrature point
 * [In] filter_q : boolean
 *          indicator of whether the discharge is filtered.
 * [In] t : double
 *          current time
 * [In] Tlast : double 
 *          the final time.
 * [Out] rhs : double array
 *          rhs is the right-hand-side constructed using the input state u.
 * [Out] dt : double
 *          time step
*/
void rhs1d_cu_es1(double *u, double *bb, double *trip_prod, double dx, \
  int nx, int K_pce, double grav, double *eval_point, \
  int num_point, bool filter_q, double t, double Tlast, \
  double *dt, double *rhs)
{

  int i, j, k;
  double *h, *uu, *htemp, *rhs_temp, cfl;
  double *Hx, *source, eps = pow(dx, 4), *amax, am, *e_var;
	//clock_t start, end;
	//double t_filter, t_speeds, t_veld, t_flux1, t_globalt;

  // Memory allocation 

  uu = malloc((nx + 2) * (K_pce) * sizeof(double)); // PCE vector for cell average of the velocity.
  h = malloc((nx + 2) * (K_pce) * sizeof(double)); // PCE vector for cell average of the water height.
  rhs_temp = malloc((K_pce) * sizeof(double)); 
  htemp = malloc((K_pce) * sizeof(double)); 
  Hx = malloc((nx + 2) * (K_pce * 2) * sizeof(double)); // numerical flux
  e_var = malloc((nx + 2) * (K_pce * 2) * sizeof(double)); // PCE vector for the cell average of the entropy variable        
  source = malloc((nx + 2) * (K_pce * 2) * sizeof(double)); // energy-conservative source.
  amax = malloc((nx + 2) * sizeof(double));  // maximum local propagation speeds at each cell interface.

  // compute the average water height.
  for(i = 1; i <= nx; i++)
  {
    for(j = 0; j <= K_pce; j++)
    {
      h[j + i * K_pce] = u[j + i * 2 * K_pce] - bb[j + i * K_pce];
    }
  }

   // compute the corresponding PCE vector for the velocity (velocity desingularization is applied here).
  for(i = 1; i <= nx; i++)
  {
    vel_desing1d(eps, i, K_pce, h, u, trip_prod, uu);
  }

  // compute the corresponding entropy variable.
  for(i = 1; i <= nx; i++)
  {
    entropy_var(i, K_pce, uu, h, u, trip_prod, bb, grav, e_var);
  }  

  // Outflow boundaries
  for(j = 0; j < K_pce; j++)
  {
    // velocity
    uu[j] = uu[j + K_pce];
    uu[j + (nx + 1) * K_pce] = uu[j + (nx) * K_pce];
    
    // water height
    h[j] = h[j + K_pce];
    h[j + (nx + 1) * K_pce] = h[j + (nx) * K_pce];
    
    // state variables
    u[j] = u[j + 2 * K_pce];
    u[j + (nx + 1) * 2 * K_pce] = u[j + nx * 2 * K_pce];        
    u[j + K_pce] = u[j + K_pce + 2 * K_pce];
    u[j + K_pce + (nx + 1) * 2 * K_pce] = u[j + K_pce + (nx) * 2 * K_pce];   

    // entropy variable
    e_var[j] = e_var[j + 2 * K_pce];
    e_var[j + (nx + 1) * 2 * K_pce] = e_var[j + nx * 2 * K_pce];                
    e_var[j + K_pce] = e_var[j + K_pce + 2 * K_pce];
    e_var[j + K_pce + (nx + 1) * 2 * K_pce] = e_var[j + K_pce + (nx) * 2 * K_pce];   
  }

  // compute the energy conservative source
  for(i = 1; i < nx+2; i++)
  {
    source_ec_1d(i, K_pce, h, bb, dx, grav, trip_prod, source);
  }

  // compute the second-order numerical flux
  for(i = 0; i < nx+1; i++)
  {
    flux_es_1d_order1(i, K_pce, uu,  h, u, trip_prod, bb, e_var, grav, Hx, amax);
  }

  // construct the right-hand-side and compute the hyperbolicity-preserving time step.
  cfl = 0.0;
  for(i = 1; i <= nx; i++)
  {
    for(j = 0; j < 2 * K_pce; j++)
    {
      rhs[j + i * 2 * K_pce] = -(Hx[j + 2 * i * K_pce] - Hx[j + 2 * (i - 1) * K_pce])/dx + source[j + 2 * i * K_pce];
    }    
    for(k = 0; k < num_point; k++)
    {
      for(j = 0; j < K_pce; j++)
      {
        htemp[j] = h[j + i * K_pce] * \
          eval_point[k + j * num_point];
          
        rhs_temp[j] = rhs[j + i * 2 * K_pce] * \
          eval_point[k + j * num_point];
      }      
      if(array_sum(htemp, 0, K_pce) <= -h_thresh)
      {
        printf("cell %d\n", i);
        for(j = 0; j < K_pce; j++)
        {
          printf("%.16f %.16f\n", u[j+i*2*K_pce],bb[j+i*K_pce]);
        }
      }
      assert(array_sum(htemp, 0, K_pce) > -h_thresh);
      
      if(array_sum(rhs_temp, 0, K_pce) < -h_thresh)
      {
        cfl = fmax(cfl, -array_sum(rhs_temp, 0, K_pce)/\
          array_sum(htemp, 0 , K_pce));       
      }
    }     
  }
  // incorporate the hyperbolicity-preserving time step and wave-speed CFL time step
  am = array_max(amax, 1, nx + 1);
  (*dt) = 0.5 * dx / am;
  (*dt) = 0.9 * fmin((*dt), 1.0/cfl);  
  
  if(t + (*dt) > Tlast)
  {
    (*dt) = Tlast - t;
  }
  
  // Free space 
  free((void *)amax); free((void*)e_var);     
  free((void *)h); free((void *)uu); 
  free((void *)Hx); free((void *)source);
  free((void *)rhs_temp); free((void *)htemp);
}

/*
 * entropy_var : compute the entropy variable provided the PCE of the water surface (eta = h + B) and the velocity
 * 
 * Parameters:
 * [In] i : int
 *         the cell index
 * [In] K_pce : int
 *          number of expansion terms in PCE.
 * [In] uu : double array
 *          PCE for the velocity
 * [In] h : double array
 *          PCE for the water height
 * [In] u : double array
 *          PCE for the state
 * [In] trip_prod : double array
 *          array of triple (inner) product of the orthonormal polynomials
 * [In] bb : double array
 *          cell average of the piecewise linear reconstructed bottom topography
 * [In] grav : double 
 *          gravitational constant
 * [Out] e_var: double array
 *          the corresponding entropy variable at the given cell
*/
void entropy_var(int i, int K_pce, double *uu, double *h, double *u, double *trip_prod, double *bb, double grav, double *e_var)
{
  double *P_u, *u_local, *v1;
  int j, incxy = 1;
  double zero = 0.0, one = 1.0;
      
  P_u = malloc(K_pce * K_pce * sizeof(double));
  u_local = malloc(K_pce * sizeof(double));
  v1 = malloc(K_pce * sizeof(double));
  
  for(j = 0; j < K_pce; j++)
  {
    u_local[j] = uu[j + i * K_pce];
  }
  pce_matrix(trip_prod, K_pce, u_local, P_u); // compute the matrix P(u)

  dsymv_("U", &K_pce, &one, P_u, &K_pce, u_local, &incxy, &zero, v1, &incxy); // compute P(u)u
  for(j = 0; j < K_pce; j++)
  {
    e_var[j + 2*i*K_pce] = -0.5*v1[j]+grav*u[j + i*2*K_pce]; // first K_pce rows of the entropy variable
    e_var[(j+K_pce) + 2*i*K_pce] = uu[j+i*K_pce]; // last K_pce rows of the entropy variable
  }
  free(P_u);
  free(u_local); free(v1);
}

/*
 * flux_es_1d_order2 : compute the second-order energy-stable numerical flux at cell interface i+1/2.
 *
 * Parameters:
 * [In] i : int
 *          the index of the cell
 * [In] K_pce : int
 *          number of expansion terms in PCE.
 * [In] uu : double array
 *          the PCE of the average velocity, which is used in constructing the energy conservative part of the flux
 * [In] uuE : double array
 *          east-side reconstruction of the PCE of the velocity
 * [In] uuW : double array
 *          west-side reconstruction of the PCE of the velocity
 * [In] h : double array
 *          the PCE of the average water height, which is used in constructing the energy conservative part of the flux
 * [In] hE : double array
 *          east-side reconstruction of the PCE of the water height
 * [In] hW : double array
 *          west-side reconstruction of the PCE of the water height
 * [In] trip_prod : double array
 *          array of triple (inner) product of the orthonormal polynomials
 * [In] dx : double
 *          (uniform) spatial grid size
 * [In] wtp : double array
 *          east-side reconstruction of the second-order scaled variable
 * [In] wtm : double arrayy
 *          west-side reconstruction of the second-order scaled variable
 * [In] grav : double 
 *          graviational constant
 * [Out] f : double array
 *          the array of numerical fluxes at interfaces.
 * [Out] amax : double array
 *          maximum wave-speed at the interface
*/
void flux_es_1d_order2(int i, int K_pce, double *uu, double *uuE, double *uuW, \
double *h, double *hE, double *hW, double *trip_prod, double dx, \
double *wtp, double *wtm, double grav, double *f, double *amax)
{
  double *P_uleft, *P_uright, *P_uavg, *P_havg, *P_qavg, *P_hsqrt, *P_havgcopy;
  double *P_hsqrtinv, *B2, *B3;
  double *wtjump, *uu_avg, *h_avg, *q_avg;
  double *R, *D, *T, *difv;
  int info, dim = 2 * K_pce;  
  double *work, w[dim], wkopt, zero = 0.0, one = 1.0;
  int incxy = 1;  
  int j, k, m;

  uu_avg = malloc(K_pce * sizeof(double)); // average velocity
  h_avg = malloc(K_pce * sizeof(double)); // average height h_avg
  q_avg = malloc(K_pce * sizeof(double)); // "average" discharge defined by q_avg = P(h_avg)u_avg.

  P_havg = malloc(K_pce * K_pce * sizeof(double)); // the matrix gP(h_avg)
  P_havgcopy = malloc(K_pce * K_pce * sizeof(double)); // the matrix P(h_avg)
  P_uavg = malloc(K_pce * K_pce * sizeof(double)); // the matrix P(u_avg)
  P_qavg = malloc(K_pce * K_pce * sizeof(double));  // the matrix P(q_avg)
  P_hsqrt = malloc(K_pce * K_pce * sizeof(double)); // the square root matrix sqrt{gP(h_avg)}
  P_hsqrtinv = malloc(K_pce * K_pce * sizeof(double)); // the inverse of the square root matrix (sqrt{gP(h_avg)})^{-1}

  B2 = malloc(K_pce * K_pce * sizeof(double)); // g(gP(h_avg))^{-1}P(q_avg)(gP(h_avg))^{-1}
  B3 = malloc(K_pce * K_pce * sizeof(double)); // auxillary matrix

  R = malloc(4 * K_pce * K_pce * sizeof(double)); // R matrix
  T = malloc(4 * K_pce * K_pce * sizeof(double)); // the eigenmatrix constructed using the R matrix
  D = malloc(4 * K_pce * K_pce * sizeof(double)); // the matrix after symmetrization

  wtjump = malloc(2 * K_pce * sizeof(double));
  difv = malloc(2 * K_pce * sizeof(double));

  // compute the jumps on the scaled entropic variables
  for(j = 0; j < dim; j++)
  {
    wtjump[j] = wtm[j + (i + 1) * dim] - wtp[j + i * dim];
  }
  //printf("%d %.16f %.16f\n", i, wtjump[0], wtjump[K_pce]);
  // assembling the Jacobian matrix evaluated at the average state
  for(j = 0; j < K_pce; j++)
  {
    h_avg[j] = 0.5*(hE[j + i * K_pce] + hW[j + (i + 1) * K_pce]);
    uu_avg[j] = 0.5*(uuE[j + i * K_pce] + uuW[j + (i + 1) * K_pce]);
  }
  
  pce_matrix(trip_prod, K_pce, h_avg, P_havg);
  pce_matrix(trip_prod, K_pce, uu_avg, P_uavg);
  fcopy_array(K_pce*K_pce, P_havg, P_havgcopy);
  for(j = 0; j < K_pce * K_pce; j++)
  {
    P_havg[j] *= grav;
  }
  dsymv_("U", &K_pce, &one, P_havgcopy, &K_pce, uu_avg, &incxy, &zero, q_avg, &incxy);
  pce_matrix(trip_prod, K_pce, q_avg, P_qavg);
  sqrt_matrix(P_havg, K_pce, P_hsqrt); // sqrt{gP(h)}

  // compute the matrix R that symmetrize the original Jacobian (note that LAPACK is column-major order)
  for(j = 0; j < K_pce; j++)
  {
    for(k = 0; k < K_pce; k++)
    {
      if(j == k)
      {
        R[k + j * 2 * K_pce] = 1.0/sqrt(2*grav);
        R[k + (j + K_pce) * 2 * K_pce] = 1.0/sqrt(2*grav);
      }
      else
      {
        R[k + j * 2 * K_pce] = 0;
        R[k + (j + K_pce) * 2 * K_pce] = 0;
      }
      R[(k + K_pce) + j * 2 * K_pce] = 1.0/sqrt(2*grav) * \
        (P_uavg[j + k * K_pce] + P_hsqrt[j + k * K_pce]);
      R[(k + K_pce) + (j + K_pce) * 2 * K_pce] = 1.0/sqrt(2*grav) * \
        (P_uavg[j + k * K_pce] - P_hsqrt[j + k * K_pce]);      
    }
  }
  assert(h_avg[0] > -h_thresh);  

  // compute sqrt{gP(h_avg)}^{-1}
  for(j = 0; j < K_pce; j++)
  {
    for(k = 0; k < K_pce; k++)
    {
      P_hsqrtinv[k + j * K_pce] = P_hsqrt[k + j * K_pce];
    }
  }    
  inverse(K_pce, P_hsqrtinv);
  
  // compute g(gP(h_avg))^{-1}P(q_avg)(gP(h_avg))^{-1}
  dgemm_("N","N",&K_pce, &K_pce, &K_pce, &one, P_hsqrtinv, &K_pce, P_qavg,
    &K_pce, &zero, B2, &K_pce); 
  dgemm_("N","N",&K_pce, &K_pce, &K_pce, &one, B2, &K_pce, P_hsqrtinv,
    &K_pce, &zero, B3, &K_pce); 
  for(j = 0; j < K_pce; j++)
  {
    for(k = 0; k < K_pce; k++)
    {
      B2[k + j *K_pce] = 0.5 * grav * (B3[k + j * K_pce] + B3[j + k * K_pce]); // ensure symmetry 
    }
  }

  // construct the symmetric matrix after symmetrization 
  for(j = 0; j < K_pce; j++)
  {
    for(k = 0; k < K_pce; k++)
    {
      D[k + j * 2 * K_pce] = 0.5 * (2 * P_hsqrt[j + k * K_pce] + \
        P_uavg[j + k * K_pce] + B2[j + k * K_pce]); 
      D[(k + K_pce) + j * 2 * K_pce] = 0.5 * (P_uavg[j + k * K_pce] - B2[j + k * K_pce]);
      D[k + (j + K_pce) * 2 * K_pce] = 0.5 * (P_uavg[j + k * K_pce] - B2[j + k * K_pce]);
      D[(k + K_pce) + (j + K_pce) * 2 * K_pce] = 0.5 * (-2 * P_hsqrt[j + k * K_pce] + \
        P_uavg[j + k * K_pce] + B2[j + k * K_pce]);
    }
  }
  sym_eigen_decom(dim, D, w); // eigendecomposition of D, output D is the eigenmatrix and w is the vector of eigenvalues.
	dgemm_("N","N",&dim, &dim, &dim, &one, R, &dim, D,
		&dim, &zero, T, &dim);  

  for(j = 0; j < dim; j++)
  {
    wtjump[j] *= fabs(w[j]); // scaled the scale variable by the absolute values of the eigenvalues.    
  }
  
  amax[i] = array_max(w, 0, dim);  // store the maximum wave speed at the cell interface

  dsymv_("U", &dim, &one, T, &dim, wtjump, &incxy, &zero, difv, &incxy); // compute the diffusion term
  flux_ec_1d(i, K_pce, uu, h, trip_prod, grav, f); // EC flux
  for(j = 0 ; j < dim; j++)
  {
    f[j + i * dim] -= 0.5 * difv[j]; // combine the EC flux and the diffusion term
  }

  free(T); free(wtjump);
  free(h_avg); free(uu_avg); free(q_avg);
  free(P_havg); free(P_uavg); free(P_qavg); free(P_hsqrt); 
  free(P_hsqrtinv); free(B2); free(B3); free(P_havgcopy);
  free(R); free(D); free(difv);
}

/*
 * scale_var_order1 : compute the first-order scaled entropy variable for the i-th cell
 *
 * Parameters:
 * [In] i : int
 *          the index of the cell
 * [In] K_pce : int
 *          number of expansion terms in PCE.
 * [In] uu : double array
 *          the PCE of the average velocity, which is used in constructing the energy conservative part of the flux
 * [In] uuE : double array
 *          east-side reconstruction of the PCE of the velocity
 * [In] uuW : double array
 *          west-side reconstruction of the PCE of the velocity
 * [In] h : double array
 *          the PCE of the average water height, which is used in constructing the energy conservative part of the flux
 * [In] hE : double array
 *          east-side reconstruction of the PCE of the water height
 * [In] hW : double array
 *          west-side reconstruction of the PCE of the water height
 * [In] trip_prod : double array
 *          array of triple (inner) product of the orthonormal polynomials
 * [In] dx : double
 *          (uniform) spatial grid size
 * [Out] wp : double array
 *          east-side reconstruction of the first-order scaled variable
 * [Out] wm : double arrayy
 *          west-side reconstruction of the first-order scaled variable
 * [In] ent : double array
 *          cell average of the entropy variable
 * [In] grav : double 
 *          graviational constant
*/
void scale_var_order1(int i, int K_pce, double *uu, double *uuE, double *uuW, \
double *h, double *hE, double *hW, double *trip_prod, \
double *wm, double *wp, double *bb, double dx, double *ent, double grav)
{
  double *P_uavg, *P_havg, *P_qavg, *P_hsqrt, *P_havgcopy;
  double *P_hsqrtinv, *B2, *B3;
  double *uu_avg, *h_avg, *q_avg, *wp_local, *wm_local;
  double *R, *L, *D, *T, *T_tran, *ent_local, *ent_local2;
  int lwork = -1, info, dim = 2 * K_pce;  
  double *work, w[dim], wkopt, zero = 0.0, one = 1.0;
  int incxy = 1;  
  int j, k, m;
  P_havg = malloc(K_pce * K_pce * sizeof(double));
  P_havgcopy = malloc(K_pce * K_pce * sizeof(double));
  P_uavg = malloc(K_pce * K_pce * sizeof(double));
  P_qavg = malloc(K_pce * K_pce * sizeof(double));
  P_hsqrt = malloc(K_pce * K_pce * sizeof(double));
  P_hsqrtinv = malloc(K_pce * K_pce * sizeof(double));
  B2 = malloc(K_pce * K_pce * sizeof(double));
  B3 = malloc(K_pce * K_pce * sizeof(double));
  R = malloc(4 * K_pce * K_pce * sizeof(double));
  T = malloc(4 * K_pce * K_pce * sizeof(double));
  T_tran = malloc(4 * K_pce * K_pce * sizeof(double));

  L = malloc(4 * K_pce * K_pce * sizeof(double));
  D = malloc(4 * K_pce * K_pce * sizeof(double));

  uu_avg = malloc(K_pce * sizeof(double)); // PCE for the average velocity
  h_avg = malloc(K_pce * sizeof(double)); // PCE for the average height
  q_avg = malloc(K_pce * sizeof(double)); // PCE for the average discharge
  wp_local = malloc(2 * K_pce * sizeof(double)); // local variable for east-side reconstruction of the scaled entropic variable
  wm_local = malloc(2 * K_pce * sizeof(double)); // local variable for west-side reconstruction of the scaled entropic variable
  ent_local = malloc(2 * K_pce * sizeof(double)); // entropy variable in the i-th cell
  ent_local2 = malloc(2 * K_pce * sizeof(double)); // entropy variable in the (i+1)-th cell

  for(j = 0; j < K_pce; j++)
  {
    h_avg[j] = 0.5*(hE[j + i * K_pce] + hW[j + (i + 1) * K_pce]);
    uu_avg[j] = 0.5*(uuE[j + i * K_pce] + uuW[j + (i + 1) * K_pce]);
    ent_local[j] = ent[j + i * 2 * K_pce];
    ent_local[j + K_pce] = ent[j + K_pce + i * 2 * K_pce];
    ent_local2[j] = ent[j + (i+1) * 2 * K_pce];
    ent_local2[j + K_pce] = ent[j + K_pce + (i+1) * 2 * K_pce];    
  }
  
  pce_matrix(trip_prod, K_pce, h_avg, P_havg);
  pce_matrix(trip_prod, K_pce, uu_avg, P_uavg);
  fcopy_array(K_pce*K_pce, P_havg, P_havgcopy);
  for(j = 0; j < K_pce * K_pce; j++)
  {
    P_havg[j] *= grav;
  }
  dsymv_("U", &K_pce, &one, P_havgcopy, &K_pce, uu_avg, &incxy, &zero, q_avg, &incxy); // compute q_avg

  pce_matrix(trip_prod, K_pce, q_avg, P_qavg);
  sqrt_matrix(P_havg, K_pce, P_hsqrt); // sqrt{gP(h)}
  //the matrix R that symmetrize the original Jacobian
  for(j = 0; j < K_pce; j++)
  {
    for(k = 0; k < K_pce; k++)
    {
      if(j == k)
      {
        R[k + j * 2 * K_pce] = 1.0/sqrt(2*grav);
        R[k + (j + K_pce) * 2 * K_pce] = 1.0/sqrt(2*grav);
      }
      else
      {
        R[k + j * 2 * K_pce] = 0;
        R[k + (j + K_pce) * 2 * K_pce] = 0;
      }
      R[(k + K_pce) + j * 2 * K_pce] = 1.0/sqrt(2*grav) * \
        (P_uavg[j + k * K_pce] + P_hsqrt[j + k * K_pce]);
      R[(k + K_pce) + (j + K_pce) * 2 * K_pce] = 1.0/sqrt(2*grav) * \
        (P_uavg[j + k * K_pce] - P_hsqrt[j + k * K_pce]);      
    }
  }

  assert(h_avg[0] > -h_thresh);

  // compute sqrt{gP(h)}^{-1}
  for(j = 0; j < K_pce; j++)
  {
    for(k = 0; k < K_pce; k++)
    {
      P_hsqrtinv[k + j * K_pce] = P_hsqrt[k + j * K_pce];
    }
  }    
  inverse(K_pce, P_hsqrtinv);

  // compute g(gP(h_avg))^{-1}P(q_avg)(gP(h_avg))^{-1}
  dgemm_("N","N",&K_pce, &K_pce, &K_pce, &one, P_hsqrtinv, &K_pce, P_qavg,
    &K_pce, &zero, B2, &K_pce); 
  dgemm_("N","N",&K_pce, &K_pce, &K_pce, &one, B2, &K_pce, P_hsqrtinv,
    &K_pce, &zero, B3, &K_pce); 
  for(j = 0; j < K_pce; j++)
  {
    for(k = 0; k < K_pce; k++)
    {
      B2[k + j *K_pce] = 0.5 * grav * (B3[k + j * K_pce] + B3[j + k * K_pce]); // ensure symmetry
    }
  }

  // the symmetric matrix after symmetrization 
  for(j = 0; j < K_pce; j++)
  {
    for(k = 0; k < K_pce; k++)
    {
      D[k + j * 2 * K_pce] = 0.5 * (2 * P_hsqrt[j + k * K_pce] + \
        P_uavg[j + k * K_pce] + B2[j + k * K_pce]); 
      D[(k + K_pce) + j * 2 * K_pce] = 0.5 * (P_uavg[j + k * K_pce] - B2[j + k * K_pce]);
      D[k + (j + K_pce) * 2 * K_pce] = 0.5 * (P_uavg[j + k * K_pce] - B2[j + k * K_pce]);
      D[(k + K_pce) + (j + K_pce) * 2 * K_pce] = 0.5 * (-2 * P_hsqrt[j + k * K_pce] + \
        P_uavg[j + k * K_pce] + B2[j + k * K_pce]);
    }
  }
  sym_eigen_decom(dim, D, w); // eigendecomposition of D

	dgemm_("N","N",&dim, &dim, &dim, &one, R, &dim, D,
		&dim, &zero, T, &dim);  // compute matrix T
  mat_tran(dim, dim, T, T_tran);
 
  dsymv_("U", &dim, &one, T_tran, &dim, ent_local, &incxy, &zero, wp_local, &incxy);
  dsymv_("U", &dim, &one, T_tran, &dim, ent_local2, &incxy, &zero, wm_local, &incxy);  
  for(j = 0; j < 2 * K_pce; j++)
  {
    wp[j + i * 2 * K_pce] = wp_local[j];
    wm[j + (i + 1) * 2 * K_pce] = wm_local[j];
  }
  free(T); free(T_tran);
  free(wp_local); free(wm_local);
  free(h_avg); free(uu_avg); free(q_avg);
  free(P_havg); free(P_uavg); free(P_qavg); free(P_hsqrt); 
  free(P_hsqrtinv); free(B2); free(B3); free(P_havgcopy);
  free(R); free(L); free(D); free(ent_local); free(ent_local2);
}

/*
 * scale_var_order2 : compute the second-order scaled entropy variable for the i-th cell
 *
 * Parameters:
 * [In] i : int
 *          the index of the cell
 * [In] K_pce : int
 *          number of expansion terms in PCE.
 * [In] wp : double array
 *          east-side reconstruction of the first-order scaled variable
 * [In] wm : double arrayy
 *          west-side reconstruction of the first-order scaled variable
 * [Out] wtp : double array
 *          east-side reconstruction of the second-order scaled variable
 * [Out] wtm : double arrayy
 *          west-side reconstruction of the second-order scaled variable
*/
void scale_var_order2(int i, int K_pce, double *wm, double *wp, double *wtp, double *wtm)
{
  double wr, wl, slx; // temporary variables for the right and the left jumps
  int j;

  for(j = 0; j < 2 * K_pce; j++)
  {
    wr = wm[j + (i + 1) * 2 * K_pce] - wp[j + i * 2 * K_pce];
    wl = wm[j + i * 2 * K_pce] - wp[j + (i - 1) * 2 * K_pce];
    slx = fminmod2(wr, wl);
    wtp[j + i * 2 * K_pce] = wp[j + i * 2 * K_pce] + 0.5 * slx;
    wtm[j + i * 2 * K_pce] = wm[j + i * 2 * K_pce] - 0.5 * slx;
  }

}

void flux_es_1d_order1(int i, int K_pce, double *uu, double *h, double *u, double *trip_prod, double *bb, double *e_var, \
double grav, double *f, double *amax)
{
  // 1. Call the energy conservative flux
  // 2. Add the diffusion
  double *P_uavg, *P_havg, *P_qavg, *P_hsqrt;
  double *P_hsqrtinv, *B2, *B3, *P_havgcopy;
  double *v1, *v2, *vjump, *uu_avg, *h_avg, *q_avg;
  double *R, *R_tran, *T, *D, *D_copy, *difv;
  int lwork = -1, info, dim = 2 * K_pce;  
  double *work, w[dim], wkopt, zero = 0.0, one = 1.0;
  int incxy = 1;  
  int j, k, m;
  P_havg = malloc(K_pce * K_pce * sizeof(double));
  P_havgcopy = malloc(K_pce * K_pce * sizeof(double));
  P_uavg = malloc(K_pce * K_pce * sizeof(double));
  P_qavg = malloc(K_pce * K_pce * sizeof(double));
  P_hsqrt = malloc(K_pce * K_pce * sizeof(double));
  P_hsqrtinv = malloc(K_pce * K_pce * sizeof(double));
  B2 = malloc(K_pce * K_pce * sizeof(double));
  B3 = malloc(K_pce * K_pce * sizeof(double));
  R = malloc(4 * K_pce * K_pce * sizeof(double));
  R_tran = malloc(4 * K_pce * K_pce * sizeof(double));
  T = malloc(4 * K_pce * K_pce * sizeof(double));
  D = malloc(4 * K_pce * K_pce * sizeof(double));
  D_copy = malloc(4 * K_pce * K_pce * sizeof(double));

  uu_avg = malloc(K_pce * sizeof(double));
  h_avg = malloc(K_pce * sizeof(double));
  q_avg = malloc(K_pce * sizeof(double));
  v1 = malloc(K_pce * sizeof(double));
  v2 = malloc(K_pce * sizeof(double));
  vjump = malloc(2 * K_pce * sizeof(double));
  difv = malloc(2 * K_pce * sizeof(double));
  // Step 1: jumps on the entropic variables 
  for(j = 0; j < dim; j++)
  {
    vjump[j] = e_var[j+(i+1)*dim] - e_var[j+i*dim];
  }

  // Step 2: Assembling the ES1 operator.
  for(j = 0; j < K_pce; j++)
  {
    h_avg[j] = 0.5*grav*(h[j + i * K_pce] + h[j + (i + 1) * K_pce]);
    uu_avg[j] = 0.5*(uu[j + i * K_pce] + uu[j + (i + 1) * K_pce]);
    q_avg[j] = 0.5*(u[(j + K_pce) + i * 2 * K_pce] + u[(j + K_pce) + (i + 1) * 2 * K_pce]);
  }
  pce_matrix(trip_prod, K_pce, h_avg, P_havg);
  pce_matrix(trip_prod, K_pce, uu_avg, P_uavg);
  fcopy_array(K_pce*K_pce, P_havg, P_havgcopy);
  //dsymv_("U", &K_pce, &one, P_havgcopy, &K_pce, uu_avg, &incxy, &zero, q_avg, &incxy);
  sqrt_matrix(P_havg, K_pce, P_hsqrt); // sqrt{gP(h)}
  // Step 2-1: matrix R
  for(j = 0; j < K_pce; j++)
  {
    for(k = 0; k < K_pce; k++)
    {
      if(j == k)
      {
        R[k + j * 2 * K_pce] = 1/sqrt(2*grav);
        R[k + (j + K_pce) * 2 * K_pce] = 1/sqrt(2*grav);
      }
      else
      {
        R[k + j * 2 * K_pce] = 0;
        R[k + (j + K_pce) * 2 * K_pce] = 0;
      }
      R[(k + K_pce) + j * 2 * K_pce] = 1/sqrt(2*grav) * \
        (P_uavg[j + k * K_pce] + P_hsqrt[j + k * K_pce]);
      R[(k + K_pce) + (j + K_pce) * 2 * K_pce] = 1/sqrt(2*grav) * \
        (P_uavg[j + k * K_pce] - P_hsqrt[j + k * K_pce]);      
    }
  }
  for(j = 0; j < dim; j++)
  {
    for(k = 0; k < dim; k++)
    {
      R_tran[k + j * dim] = R[j + k * dim];
    }
  }

  if(fabs(h_avg[0]) < h_thresh)
  {
    for(j = 0; j < K_pce; j++)
    {
      for(k = 0; k < K_pce; k++)
      {
        P_hsqrtinv[k + j * K_pce] = 0;
      }
    }    
  }
  else
  {
    for(j = 0; j < K_pce; j++)
    {
      for(k = 0; k < K_pce; k++)
      {
        P_hsqrtinv[k + j * K_pce] = P_hsqrt[k + j * K_pce];
      }
    }    
    inverse(K_pce, P_hsqrtinv);
  }
  dgemm_("N","N",&K_pce, &K_pce, &K_pce, &one, P_hsqrtinv, &K_pce, P_qavg,
    &K_pce, &zero, B2, &K_pce); 
  dgemm_("N","N",&K_pce, &K_pce, &K_pce, &one, B2, &K_pce, P_hsqrtinv,
    &K_pce, &zero, B3, &K_pce); 
  for(j = 0; j < K_pce; j++)
  {
    for(k = 0; k < K_pce; k++)
    {
      B2[k + j *K_pce] = 0.5 * grav * (B3[k + j * K_pce] + B3[j + k * K_pce]);
    }
  }
  for(j = 0; j < K_pce; j++)
  {
    for(k = 0; k < K_pce; k++)
    {
      D[k + j * 2 * K_pce] = 0.5 * (2 * P_hsqrt[j + k * K_pce] + \
        P_uavg[j + k * K_pce] + B2[j + k * K_pce]); 
      D[(k + K_pce) + j * 2 * K_pce] = 0.5 * (P_uavg[j + k * K_pce] - B2[j + k * K_pce]);
      D[k + (j + K_pce) * 2 * K_pce] = 0.5 * (P_uavg[j + k * K_pce] - B2[j + k * K_pce]);
      D[(k + K_pce) + (j + K_pce) * 2 * K_pce] = 0.5 * (-2 * P_hsqrt[j + k * K_pce] + \
        P_uavg[j + k * K_pce] + B2[j + k * K_pce]);
    }
    //D_copy = 
  }
  dsyev_("V", "U", &dim, D, &dim, w, &wkopt, &lwork, &info);

  lwork = (int)wkopt;
  work = (double*)malloc(lwork*sizeof(double));
  dsyev_("V", "U", &dim, D, &dim, w, work, &lwork, &info);
  free( (void*)work );

  for(j = 0; j < dim; j++)
  {
    w[j] = fabs(w[j]);
  }
  amax[i] = array_max(w, 0, dim);
  for(j = 0; j < dim; j++)
  {
    for(k = 0; k < dim; k++)
    {
      T[j + k * dim] = w[j] * D[k + j * dim];
    }
  }
	dgemm_("N","N",&dim, &dim, &dim, &one, D, &dim, T,
		&dim, &zero, D_copy, &dim);
	dgemm_("N","N",&dim, &dim, &dim, &one, R, &dim, D_copy,
		&dim, &zero, D, &dim);
	dgemm_("N","N",&dim, &dim, &dim, &one, D, &dim, R_tran,
		&dim, &zero, D_copy, &dim);    
  dsymv_("U", &dim, &one, D_copy, &dim, vjump, &incxy, &zero, difv, &incxy);
  flux_ec_1d(i, K_pce, uu, h, trip_prod, grav, f);
  for(j = 0 ; j < dim; j++)
  {
    //f[j + i * dim] -= 0.5 * difv[j]; 
  }
  //printf("%.16f %.16f\n", difv[0], difv[K_pce]);
  free(v1); free(v2); free(vjump);
  free(h_avg); free(uu_avg); free(q_avg);
  free(P_havg); free(P_uavg); free(P_qavg); free(P_hsqrt); 
  free(P_hsqrtinv); free(B2); free(B3); free(P_havgcopy);
  free(R); free(R_tran); free(T); free(D); free(difv);
  free(D_copy);
}

/*
 * Function: flux1d
 * ----------------
 * compute the numerical flux for 1DSGSWE in central-upwind scheme
 * 
 * [In]     i: the ith cell.
 * 
 * [In]     K_pce: the number of terms in PCE.
 * 
 * [In]     uE: the east-side pointwise reconstruction for the state variable.
 * 
 * [In]     uW: the west-side pointwise reconstruction for the state variable.
 * 
 * [In]     hE: the east-side pointwise reconstruction for the water height.
 * 
 * [In]     hW: the west-side pointwise reconstruction for the water height.
 * 
 * [In]     uuE: the east-side pointwise reconstruction for the velocity.
 * 
 * [In]     uuW: the west-side pointwise reconstruction for the velocity.
 * 
 * [In]     trip_prod: the triple product matrix.
 * 
 * [In]     grav: the gravitational constant.
 * 
 * [In]     axm: the minmum propagation speed vector.
 * 
 * [In]     axp: the maximum propagation speed vector.
 * 
 * [In/Out] f: the numerical flux.
 */

void flux_ec_1d(int i, int K_pce, double *uu, double *h, double *trip_prod, \
  double grav, double *f)
{
  
  double *h_avg;
  double *uu_avg;
  double *P_h, *P_u, *P_hleft, *P_hright;
  double *htemp1, *htemp2; 
  double *v1, *v2, *v3, *v4;
  int j, k, m, i1 = i + 1;
  double zero = 0.0, one = 1.0;
  int incxy = 1;

  htemp1 = malloc(K_pce * sizeof(double));
  htemp2 = malloc(K_pce * sizeof(double));
  v1 = malloc(K_pce * sizeof(double));
  v2 = malloc(K_pce * sizeof(double));
  v3 = malloc(K_pce * sizeof(double));
  v4 = malloc(K_pce * sizeof(double));
  h_avg = malloc(K_pce * sizeof(double));
  uu_avg = malloc(K_pce * sizeof(double));
  P_h = malloc(K_pce * K_pce * sizeof(double));
  P_u = malloc(K_pce * K_pce * sizeof(double));
  P_hleft = malloc(K_pce * K_pce * sizeof(double));
  P_hright = malloc(K_pce * K_pce * sizeof(double));


  for(j = 0; j < K_pce; j++)
  {
    h_avg[j] = 0.5 * (h[j + i * K_pce] + h[j + (i + 1) * K_pce]);   
    uu_avg[j] = 0.5 * (uu[j + i * K_pce] + uu[j + (i + 1) * K_pce]);   
  }  
  
  // left flux
  pce_matrix(trip_prod, K_pce, h_avg, P_h);
  pce_matrix(trip_prod, K_pce, uu_avg, P_u);
  for(j = 0; j < K_pce; j++)
  {
    htemp1[j] = h[j + i * K_pce];
  }
  pce_matrix(trip_prod, K_pce, htemp1, P_hleft);
  for(j = 0; j < K_pce; j++)
  {
    htemp2[j] = h[j + (i + 1) * K_pce];
  }  
  pce_matrix(trip_prod, K_pce, htemp2, P_hright);

  dsymv_("U", &K_pce, &one, P_h, &K_pce, uu_avg, &incxy, &zero, v1, &incxy);
  dsymv_("U", &K_pce, &one, P_u, &K_pce, v1, &incxy, &zero, v2, &incxy);
  dsymv_("U", &K_pce, &one, P_hleft, &K_pce, htemp1, &incxy, &zero, v3, &incxy);
  dsymv_("U", &K_pce, &one, P_hright, &K_pce, htemp2, &incxy, &zero, v4, &incxy);
  for(j = 0; j < K_pce; j++)
  {
    f[j + i * 2 * K_pce] = v1[j]; 
    f[(j + K_pce) + i * 2 * K_pce] = 0.5*grav*0.5*(v3[j]+v4[j]) + v2[j];
  }
  //f[1 + i * 2 * K_pce] = 0;
  //f[1 + K_pce + i * 2 * K_pce] = 0; 
  //f[i * 2 * K_pce] = 0.0;
  //f[i * 2 * K_pce] = h_avg[0]*uu_avg[0];
  //f[K_pce + i * 2 * K_pce] = 0.5 * grav * (0.5 * (h[i * K_pce] * h[i * K_pce]\
   + h[(i + 1) * K_pce] * h[(i + 1) * K_pce])) + h_avg[0]*uu_avg[0]*uu_avg[0];
  //printf("%.16f %.16f %.16f\n", htemp1[0], htemp2[0],)
  free(h_avg); free(uu_avg);
  free(P_h); free(P_u); free(P_hleft); free(P_hright);
  free(v1); free(v2); free(v3); free(v4);
  free(htemp1); free(htemp2); 
}

/*
 * Function: flux1d
 * ----------------
 * compute the numerical flux for 1DSGSWE in central-upwind scheme
 * 
 * [In]     i: the ith cell.
 * 
 * [In]     K_pce: the number of terms in PCE.
 * 
 * [In]     uE: the east-side pointwise reconstruction for the state variable.
 * 
 * [In]     uW: the west-side pointwise reconstruction for the state variable.
 * 
 * [In]     hE: the east-side pointwise reconstruction for the water height.
 * 
 * [In]     hW: the west-side pointwise reconstruction for the water height.
 * 
 * [In]     uuE: the east-side pointwise reconstruction for the velocity.
 * 
 * [In]     uuW: the west-side pointwise reconstruction for the velocity.
 * 
 * [In]     trip_prod: the triple product matrix.
 * 
 * [In]     grav: the gravitational constant.
 * 
 * [In]     axm: the minmum propagation speed vector.
 * 
 * [In]     axp: the maximum propagation speed vector.
 * 
 * [In/Out] f: the numerical flux.
 */

void flux_ec_1d_es2(int i, int K_pce, double *uuE, double *uuW, double *hE, double *hW, double *trip_prod, \
  double grav, double *f)
{
  
  double *h_avg;
  double *uu_avg;
  double *P_h, *P_u, *P_hleft, *P_hright;
  double *htemp1, *htemp2; 
  double *v1, *v2, *v3, *v4, *v5;
  int j, k, m, i1 = i + 1;
  double zero = 0.0, one = 1.0;
  int incxy = 1;

  htemp1 = malloc(K_pce * sizeof(double));
  htemp2 = malloc(K_pce * sizeof(double));
  v1 = malloc(K_pce * sizeof(double));
  v2 = malloc(K_pce * sizeof(double));
  v3 = malloc(K_pce * sizeof(double));
  v4 = malloc(K_pce * sizeof(double));
  v5 = malloc(K_pce * sizeof(double));
  h_avg = malloc(K_pce * sizeof(double));
  uu_avg = malloc(K_pce * sizeof(double));
  P_h = malloc(K_pce * K_pce * sizeof(double));
  P_u = malloc(K_pce * K_pce * sizeof(double));
  P_hleft = malloc(K_pce * K_pce * sizeof(double));
  P_hright = malloc(K_pce * K_pce * sizeof(double));


  for(j = 0; j < K_pce; j++)
  {
    h_avg[j] = 0.5 * (hE[j + i * K_pce] + hW[j + (i + 1) * K_pce]);   
    uu_avg[j] = 0.5 * (uuE[j + i * K_pce] + uuW[j + (i + 1) * K_pce]);   
  }  
  
  // left flux
  pce_matrix(trip_prod, K_pce, h_avg, P_h);
  pce_matrix(trip_prod, K_pce, uu_avg, P_u);
  for(j = 0; j < K_pce; j++)
  {
    htemp1[j] = hE[j + i * K_pce];
  }
  pce_matrix(trip_prod, K_pce, htemp1, P_hleft);
  for(j = 0; j < K_pce; j++)
  {
    htemp2[j] = hW[j + (i + 1) * K_pce];
  }  
  pce_matrix(trip_prod, K_pce, htemp2, P_hright);

  dsymv_("U", &K_pce, &one, P_h, &K_pce, uu_avg, &incxy, &zero, v1, &incxy);
  fcopy_array(K_pce, v1, v5);
  dsymv_("U", &K_pce, &one, P_u, &K_pce, v5, &incxy, &zero, v2, &incxy);
  dsymv_("U", &K_pce, &one, P_hleft, &K_pce, htemp1, &incxy, &zero, v3, &incxy);
  dsymv_("U", &K_pce, &one, P_hright, &K_pce, htemp2, &incxy, &zero, v4, &incxy);
  for(j = 0; j < K_pce; j++)
  {
    f[j + i * 2 * K_pce] = v1[j]; 
    f[(j + K_pce) + i * 2 * K_pce] = 0.5*grav*0.5*(v3[j]+v4[j]) + v2[j];
  }
  //f[1 + i * 2 * K_pce] = 0;
  //f[1 + K_pce + i * 2 * K_pce] = 0; 
  //f[i * 2 * K_pce] = 0.0;
  //f[i * 2 * K_pce] = h_avg[0]*uu_avg[0];
  //f[K_pce + i * 2 * K_pce] = 0.5 * grav * (0.5 * (h[i * K_pce] * h[i * K_pce]\
   + h[(i + 1) * K_pce] * h[(i + 1) * K_pce])) + h_avg[0]*uu_avg[0]*uu_avg[0];
  //printf("%.16f %.16f %.16f\n", htemp1[0], htemp2[0],)
  free(h_avg); free(uu_avg);
  free(P_h); free(P_u); free(P_hleft); free(P_hright);
  free(v1); free(v2); free(v3); free(v4); free(v5);
  free(htemp1); free(htemp2); 
}


/*
 * Function: source1d
 * --------------------
 * compute the source term in 1DSGSWE for the central upwind scheme
 * 
 * [In]     i: the ith cell.
 * 
 * [In]     K_pce: the number of term in PCE.
 * 
 * [In]     u: the cell averages of the state variables.
 * 
 * [In]     bb: the cell average of the bottom topography.
 * 
 * [In]     dx: the grid size.
 * 
 * [In]     grav: the gravitational constants.
 * 
 * [In]     trip_prod: the triple product matrix.
 * 
 * [In/Out] source: the source term.
 */

void source_ec_1d(int i, int K_pce, double *h, double *bb, \
  double dx, double grav, double *trip_prod, double *source)
{
  int j, k, m;
  double *h_leftavg, *Bx_leftjump;
  double *h_rightavg, *Bx_rightjump;
  double *P_hleft, *P_hright;
  double *v1, *v2;
  double zero = 0.0, one = 1.0;
  int incxy = 1;  
  h_leftavg = malloc(K_pce * sizeof(double));
  Bx_leftjump = malloc(K_pce * sizeof(double));
  h_rightavg = malloc(K_pce * sizeof(double));
  Bx_rightjump = malloc(K_pce * sizeof(double));
  P_hleft = malloc(K_pce * K_pce * sizeof(double));
  P_hright = malloc(K_pce * K_pce * sizeof(double));
  v1 = malloc(K_pce * sizeof(double));
  v2 = malloc(K_pce * sizeof(double));

  for(j = 0; j < K_pce; j++)
  {
    h_leftavg[j] = 0.5 * (h[j + i * K_pce] + h[j + (i - 1) * K_pce]);
    h_rightavg[j] = 0.5 * (h[j + i * K_pce] + h[j + (i + 1) * K_pce]);   

    Bx_leftjump[j] = bb[j + i * K_pce] - bb[j + (i - 1) * K_pce];
    Bx_rightjump[j] = bb[j + (i + 1) * K_pce] - bb[j + i * K_pce];
  }
  pce_matrix(trip_prod, K_pce, h_leftavg, P_hleft);  
  pce_matrix(trip_prod, K_pce, h_rightavg, P_hright);  
  dsymv_("U", &K_pce, &one, P_hleft, &K_pce, Bx_leftjump, &incxy, &zero, v1, &incxy);
  dsymv_("U", &K_pce, &one, P_hright, &K_pce, Bx_rightjump, &incxy, &zero, v2, &incxy);

  for(j = 0; j < K_pce; j++)
  {
    source[j + i * 2 * K_pce] = 0.0;
    source[(j + K_pce) + i * 2 * K_pce] = -grav/(2*dx)*(v1[j] + v2[j]);
  }
  free(v1); free(v2);
  free(P_hleft); free(P_hright);
  free(h_leftavg); free(Bx_leftjump);
  free(h_rightavg); free(Bx_rightjump);
}

/*
 * Function: source1d
 * --------------------
 * compute the source term in 1DSGSWE for the central upwind scheme
 * 
 * [In]     i: the ith cell.
 * 
 * [In]     K_pce: the number of term in PCE.
 * 
 * [In]     u: the cell averages of the state variables.
 * 
 * [In]     bb: the cell average of the bottom topography.
 * 
 * [In]     dx: the grid size.
 * 
 * [In]     grav: the gravitational constants.
 * 
 * [In]     trip_prod: the triple product matrix.
 * 
 * [In/Out] source: the source term.
 */

void source_ec_1d_es2(int i, int K_pce, double *hE, double *hW, double *bE, double *bW,\
  double dx, double grav, double *trip_prod, double *source)
{
  int j, k, m;
  double *h_leftavg, *Bx_leftjump;
  double *h_rightavg, *Bx_rightjump;
  double *P_hleft, *P_hright;
  double *v1, *v2;
  double zero = 0.0, one = 1.0;
  int incxy = 1;  
  h_leftavg = malloc(K_pce * sizeof(double));
  Bx_leftjump = malloc(K_pce * sizeof(double));
  h_rightavg = malloc(K_pce * sizeof(double));
  Bx_rightjump = malloc(K_pce * sizeof(double));
  P_hleft = malloc(K_pce * K_pce * sizeof(double));
  P_hright = malloc(K_pce * K_pce * sizeof(double));
  v1 = malloc(K_pce * sizeof(double));
  v2 = malloc(K_pce * sizeof(double));

  for(j = 0; j < K_pce; j++)
  {
    h_leftavg[j] = 0.5 * (hW[j + i * K_pce] + hE[j + (i - 1) * K_pce]);
    h_rightavg[j] = 0.5 * (hE[j + i * K_pce] + hW[j + (i + 1) * K_pce]);   

    Bx_leftjump[j] = bW[j + i * K_pce] - bE[j + (i - 1) * K_pce];
    Bx_rightjump[j] = bW[j + (i + 1) * K_pce] - bE[j + i * K_pce];
  }
  pce_matrix(trip_prod, K_pce, h_leftavg, P_hleft);  
  pce_matrix(trip_prod, K_pce, h_rightavg, P_hright);  
  dsymv_("U", &K_pce, &one, P_hleft, &K_pce, Bx_leftjump, &incxy, &zero, v1, &incxy);
  dsymv_("U", &K_pce, &one, P_hright, &K_pce, Bx_rightjump, &incxy, &zero, v2, &incxy);

  for(j = 0; j < K_pce; j++)
  {
    source[j + i * 2 * K_pce] = 0.0;
    source[(j + K_pce) + i * 2 * K_pce] = -grav/(2*dx)*(v1[j] + v2[j]);
  }
  free(v1); free(v2);
  free(P_hleft); free(P_hright);
  free(h_leftavg); free(Bx_leftjump);
  free(h_rightavg); free(Bx_rightjump);
}


void point_recon1d(double *var, int i, int dim, double *varE, \
  double *varW)
{

  int j;
  double wl, wr, slx;
  for(j = 0; j < dim; j++)
  {
    /* The left-sided, central, and right-sided difference, respectively */

    wl = var[j + i * dim] - var[j + (i - 1) * dim]; 
    wr = var[j + (i + 1) * dim] - var[j + i * dim];
 
    /* Calculate the minmod(th * wl, ux, th * wr) */
    slx = fminmod2(wl, wr);

    /* Calculate the pointwise reconstruction */
    varE[j + i * dim] = var[j + i * dim] + 0.5 * slx;
    varW[j + i * dim] = var[j + i * dim] - 0.5 * slx;
  }

}

void point_recon1d_2(double *var, int i, int dim, double *varE, \
  double *varW)
{

  int j;
  double wl, ux, wr, slx;
  for(j = 0; j < dim; j++)
  {
    /* The left-sided, central, and right-sided difference, respectively */

    wl = var[j + i * dim] - var[j + (i - 1) * dim]; 
    ux = 0.5*(var[j + (i + 1) * dim] - var[j + (i - 1) * dim]); 
    wr = var[j + (i + 1) * dim] - var[j + i * dim];
 
    /* Calculate the minmod(th * wl, ux, th * wr) */
    slx = fminmod(wl, ux, wr, 1);

    /* Calculate the pointwise reconstruction */
    varE[j + i * dim] = var[j + i * dim] + 0.5 * slx;
    varW[j + i * dim] = var[j + i * dim] - 0.5 * slx;
  }

}

void vel_desing1d(double eps, int i, int K_pce, double *h, \
  double *u, double *trip_prod, double *vx)
{
  double *P_h, *P_cor, *P_corcopy, *P_hcopy, *v, *v_copy;
  int k;
  double zero = 0.0, one = 1.0;
  int incxy = 1;

  P_h = malloc(K_pce * K_pce * sizeof(double));
  P_hcopy = malloc(K_pce * K_pce * sizeof(double));
  P_cor = malloc(K_pce * K_pce * sizeof(double));
  P_corcopy = malloc(K_pce * K_pce * sizeof(double));
  v = malloc(K_pce * sizeof(double));
  v_copy = malloc(K_pce * sizeof(double));

  /* For x-direction */

  for(k = 0; k < K_pce; k++)
  {
    v[k] = h[k + i * K_pce];
  }

  /* Step 1: compute the corrected inverse matrix. */
  pce_matrix(trip_prod, K_pce, v, P_h);
  fcopy_array(K_pce * K_pce, P_h, P_hcopy);
  
  correct_inv(eps, K_pce, P_h, P_cor);
  fcopy_array(K_pce * K_pce, P_cor, P_corcopy);
  
  /* Construct the pce vector for \hat{q_x}*/
  
  for(k = 0; k < K_pce; k++)
  {
    v_copy[k] = u[(k + K_pce) + i * 2 * K_pce];
  }

  /* Step 2: correct PCE vector for velocity. */

  dsymv_("U", &K_pce, &one, P_cor, &K_pce, v_copy, &incxy, &zero, v, &incxy);

  for(k = 0; k < K_pce; k++)
  {
    vx[k + i * K_pce] = v[k];
  }

  /* Step 3: reconstruct PCE vector for discharges */

  fcopy_array(K_pce*K_pce, P_hcopy, P_h);
  dsymv_("U", &K_pce, &one, P_h, &K_pce, v, &incxy, &zero, v_copy, &incxy);

  for(k = 0; k < K_pce; k++)
  {
    u[(k + K_pce) + i * 2 * K_pce] = v_copy[k];
  }  
  
  free((void *)P_h); free((void *)P_hcopy); 
  free((void *)P_cor); free((void *)P_corcopy); 
  free((void *)v); free((void *)v_copy);
}

/*
 * sqrt_matrix: compute the square root matrix.
 * 
 * Parameters:
 * [In] A : double array
 *          the symmetirc matrix to be taken sqrt. 
 * [In] n : int 
 *          the dimension of the mamtrix
 * [Out] A_sqrt : double array
 *          the square root matrix
*/
void sqrt_matrix(double *A, int n, double *A_sqrt)
{
  double *A_copy, *Q_tran;
  int i, j;
  double *work, w[n], wkopt, zero = 0.0, one = 1.0;
  int lwork = -1, info;

  A_copy = malloc(n * n * sizeof(double));
  Q_tran = malloc(n * n * sizeof(double));

  fcopy_array(n * n, A, A_copy);

  /* 
   * Step 1: eigenvalue decomposition of P_h. At the end of this step, the 
   * array w stores all the eigenvalues of P(\hat{h}), and the array stores 
   * all the eigenvectors of P_h (as row vectors).
   */

  dsyev_("V", "U", &n, A_copy, &n, w, &wkopt, &lwork, &info);

  lwork = (int)wkopt;
  work = (double*)malloc(lwork*sizeof(double));
  dsyev_("V", "U", &n, A_copy, &n, w, work, &lwork, &info);
  free( (void*)work );
  
  /* Step 2: reconstruct eigenvalues. */
  for(i = 0; i < n; i++)
  {
    w[i] = sqrt(w[i]);
  }

  /* 
   * Step 3: compute the matrix \Pi Q^{T} and use the array to compute
   * P_cor = Q\Pi^{cor} Q^T.
   */

  for(i = 0; i < n; i++)
  {
    for(j = 0; j < n; j++)
    {
      Q_tran[i + j * n] = w[i] * A_copy[j + i * n];
    }
  }
	dgemm_("N","N",&n, &n, &n, &one, A_copy, &n, Q_tran,
		&n, &zero, A_sqrt, &n);
  
  free(Q_tran); free(A_copy);      
}

/*
 * pce_matrix : compute the pce matrix 
 *        P(v) = \sum_{k=0}^{K_pce - 1} v_k (trip_prod)_k
 * 
 * Parameters:
 * [In]     trip_prod: the triple product tensor related to the orthonormal 
 *                   polynomial basis, where 
 *                   (trip_prod)_k = trip_prod[.,.,k] = <p_{.}p_{.}, p_k>,
 * 
 * 
 * [In]     K_pce: the number of terms in the PCE
 * 
 * [In]     v: a vector related to the PCE matrix, with length K_pce
 * 
 * [Out] P_v: the output matrix P(v)
 */
void pce_matrix(double *trip_prod, int K_pce, double *v, double *P_v)
{
  int i, j, k;
  for(i = 0; i < K_pce; i++)
  {
    for(j = 0; j < K_pce; j++)
    {
      P_v[j + i * K_pce] = 0.0;
    }
  }

  for(i = 0; i < K_pce; i++)
  {
    for(j = 0; j < K_pce; j++)
    {
      for(k = 0; k < K_pce; k++)
      {
        P_v[j + i * K_pce] += v[k] * \
          trip_prod[k + j * K_pce + i * K_pce * K_pce];
      }      
    }
  }  
}

/* 
 * correct_inv : compute the corrected inverse matrix for velocity desingularization
 * 
 * Parameters:
 * [In]     eps: tolerance. If eps = 0, this simply compute the inverse of the matrix.
 * 
 * [In]     K_pce: square matrix dimension
 * 
 * [In] P_h: original matrix
 * 
 * [Out] P_cor: corrected inverse
 */
void correct_inv(double eps, int K_pce, double *P_h, double *P_cor)
{
  double *Q_tran, *P_hcopy;
  int i, j;
  double *work, w[K_pce], wkopt, zero = 0.0, one = 1.0;
  int lwork = -1, info;

  Q_tran = malloc(K_pce * K_pce * sizeof(double));
  P_hcopy = malloc(K_pce * K_pce * sizeof(double));

  fcopy_array(K_pce * K_pce, P_h, P_hcopy);

  /* 
   * Step 1: eigenvalue decomposition of P_h. At the end of this step, the 
   * array w stores all the eigenvalues of P(\hat{h}), and the array stores 
   * all the eigenvectors of P_h (as row vectors).
   */

  dsyev_("V", "U", &K_pce, P_hcopy, &K_pce, w, &wkopt, &lwork, &info);

  lwork = (int)wkopt;
  work = (double*)malloc(lwork*sizeof(double));
  dsyev_("V", "U", &K_pce, P_hcopy, &K_pce, w, work, &lwork, &info);
  free( (void*)work );
  
  /* Step 2: reconstruct eigenvalues. */
  for(i = 0; i < K_pce; i++)
  {
    w[i] = sqrt(2.0) * w[i] / \
      (sqrt(pow(w[i], 4) + fmax(pow(w[i], 4), eps)));
    //w[i] = w[i] / (pow(w[i], 2) + eps);      
    //w[i] = 1/w[i];
  }

  /* 
   * Step 3: compute the matrix \Pi Q^{T} and use the array to compute
   * P_cor = Q\Pi^{cor} Q^T.
   */

  for(i = 0; i < K_pce; i++)
  {
    for(j = 0; j < K_pce; j++)
    {
      Q_tran[i + j * K_pce] = w[i] * P_hcopy[j + i * K_pce];
    }
  }
	dgemm_("N","N",&K_pce, &K_pce, &K_pce, &one, P_hcopy, &K_pce, Q_tran,
		&K_pce, &zero, P_cor, &K_pce);
  
  free(Q_tran); free(P_hcopy);    
}

/*
 * Function: filter_para
 * ---------------------
 * provided vec[0] > 0, this function calculate the smallest possible  
 * th in [0,1] such that,
 * th * vec[0] + (1 - th) * \sum_{i=0}^{len} vec[i] >= 0,
 * which is equivalent to,
 * vec[0] + (1 - th) * \sum_{i=1}^{len} vec[i] >= 0,
 * 
 * [In]  vec: the vector.
 * 
 * [In]  len: the length of the vector.
 * 
 * [Out] result: return the smallest possible parameter.
 */
double filter_para(double *vec, int len)
{ 
  assert(vec[0] >= 0);
  if(len == 1)
  {
    return 0;
  }
  
  double sum_vec = array_sum(vec, 1, len);

  if (sum_vec >= 0) 
  {
    return 0;
  }
  return (1.0 + vec[0]/sum_vec);
}  

/*
 * Function: ghostcell1d
 * ---------------------
 * compute the ghost cell values at the boundary using Neumann boundary condition
 * 
 * [In]     K_pce: the number of terms in PCE.
 * 
 * [In]     nx: the number of cells.
 * 
 * [In/Out] u: the state variables. On output, u[0][k] = u[1][k], and 
 *             u[nx + 1][k] = u[nx][k].
 */
void ghostcell1d(int K_pce, int nx, double *u)
{
  int j; 
/*
  for(j = 0; j < 2 * K_pce; j++)
  {
    u[j] = u[j + 2 * K_pce];
    u[j + (nx + 1) * 2 * K_pce] = u[j + nx * 2 * K_pce];
  }
  */
  for(j = 0; j < 2 * K_pce; j++)
  {
    u[j] = u[j + nx * 2 * K_pce];
    u[j + (nx + 1) * 2 * K_pce] = u[j + 2 * K_pce];
  }  
}

/*
 * Function: fminmod
 * -----------------
 * compute generalized minmod function with parameter th, 
 * minmod(th * a, b, th * c), where 
 * minmod(z_1, z_2, z_3) =
 *   (i) min(z_1, z_2, z_3), if z_1, z_2, z_3 > 0,
 *  (ii) max(z_1, z_2, z_3), if z_1, z_2, z_3 < 0,
 * (iii) 0,                  otherwise. 
 * 
 * [In]  a, b, c: quantities in the generalized minmod function
 *
 * [In]  th: the parameter in the generalized minmod function
 *
 * [Out] result: the quantity, minmod(th * a, b, th * c)
 */
double fminmod(double a, double b, double c, double th)
{
    /*
    * fminmod function returns result = minmod(th*wl, ux, th*wr)
    */
    double result = 0.0;
    a *= th;
    c *= th;
    
    if((a>0) && (b>0) && (c>0))
    {
      result = fmin(a, b);
      result = fmin(c, result);
    }
    if((a<0) && (b<0) && (c<0))
    {
      result = fmax(a, b);
      result = fmax(c, result);
    }

    return result;
}

/*
 * Function: inverse
 * -----------------
 * compute the inverse of a given matrix.
 * 
 * [In]     n: the dimension of the matrix.
 * 
 * [In/Out] P: the input matrix. On output, it is replaced by its inverse.
 */
void inverse(int n, double *P)
{
	int *IPIV;	
	int LWORK = n*n;
 	double WORK[LWORK];
	int INFO;
	IPIV = malloc(n*sizeof(int));
	dgetrf_(&n,&n,P,&n,IPIV,&INFO);
	dgetri_(&n,P,&n,IPIV,WORK,&LWORK,&INFO);
  if(INFO)
  {
    printf("failed to compute the inverse!\n");
    exit(1);
  }
}

/*
 * Function: array_min
 * -------------------
 * compute the minimum value in a given subarray
 * 
 * [In]   vec: the array.
 * 
 * [In]   start: starting position.
 * 
 * [In]   end: ending position.
 * 
 * [Out]  result: minimum value in vec[start: end - 1].
 */
double array_min(double *vec, int start, int end)
{
  double results = vec[start];
  int i;
  for(i = start + 1; i < end; i++)
  {
    if(vec[i] < results)
    {
      results = vec[i];
    }
  }
  return results;
}

/*
 * Function: array_max
 * -------------------
 * compute the maximum value in a given subarray
 * 
 * [In]   vec: the array.
 * 
 * [In]   start: starting position.
 * 
 * [In]   end: ending position.
 * 
 * [Out]  result: maximum value in vec[start: end - 1].
 */
double array_max(double *vec, int start, int end)
{
  double results = vec[start];
  int i;
  for(i = start + 1; i < end; i++)
  {
    if(vec[i] > results)
    {
      results = vec[i];
    }
  }
  return results;
}

/*
 * Function: array_abs
 * -------------------
 * output the absolute value of an array
 * 
 * 
 * [In]      start: starting position.
 * 
 * [In]      end: ending position.
 * 
 * [In/Out]  vec: the input array. On output, vec[start: end - 1] are replaced
 *                by their absolute values.
 */
void array_abs(int start, int end, double *vec)
{
  int i;
  for(i = start; i < end; i++)
  {
    vec[i] = fabs(vec[i]);
  }
}

/*
 * Function: array_sum
 * -------------------
 * compute the sum of the subarray vec[start:end-1]
 * 
 * [In]  vec: the array
 * 
 * [In]  start: starting index
 * 
 * [In]  end: ending index
 * 
 * [Out] result: the sum of the subarray vec[start:end-1]
 */
double array_sum(double *vec, int start, int end)
{
  double result = 0.0;
  int i;
  assert(end > start);
  for(i = start; i < end; i++)
  {
    result += vec[i];
  }

  return result;
}

/*
 * Function: fcopy_array
 * ----------------------
 * copy array A into B
 * 
 * [In]     len: length of the arrays.
 * 
 * [In]     A: array to be copied.
 * 
 * [In/Out] B: array to be copied into.
 */
void fcopy_array(int len, double *A, double *B)
{
  int i;
  for(i = 0; i < len; i++)
  {
    B[i] = A[i];
  }
}

/*
 * dot_prod: compute the dot product of two arrays
 *
 * Parameters:
 * [In] len : int
 *          the length of two vectors
 * [In] a : double array
 *          the first array
 * [In] b : double array
 *          the second array
 * 
 * Return
 *       a double precision value that is the dot product of a and b. 
*/
double dot_prod(int len, double *a, double *b)
{
  int i;
  double sum = 0.0;
  for(i = 0; i < len; i++)
  {
    sum += a[i] * b[i];
  }
  return sum;
}

/*
 * fminmod2 : minmod function
 *
 * Paramters: 
 * [In] a : double 
 *        the first argument
 * [In] b : doulbe
 *        the second argument
 * 
 * Return : 
 *      double value that is minmod(a, b), i.e., if a and b have the same sign, return the one with smaller
 *      magnitude, else return 0.
*/
double fminmod2(double a, double b)
{
  double result = 0.0;
  
  if((a>0) && (b>0))
  {
    result = fmin(a, b);
  }
  if((a<0) && (b<0))
  {
    result = fmax(a, b);
  }

  return result;
}
 
/*
 * sym_eigen_decom: eigendecomposition of matrix J
 * 
 * Parameters: 
 * *******************************
 * [In] dim : int
 *      the dimension of the matrix J
 * 
 * [In] J : double array
 *    On input, J is the matrix to be decomposed. On output, J contains the orthonormal eigenvectors
 *    of the original J (row-first storage).
 * 
 * [In] W : double array
 *    On output, W is the array of eigenvalues
*/
void sym_eigen_decom(int dim, double *J, double *W)
{
  int info, lwork, i, j;
  double *work, wkopt;
  double *W1;
  W1 = (double *)malloc(dim * sizeof(double));

  lwork = -1;
  dsyev_("V", "U", &dim, J, &dim, W1, &wkopt, &lwork, &info);

  lwork = (int)wkopt;
  work = (double*)malloc(lwork*sizeof(double));
  dsyev_("V", "U", &dim, J, &dim, W1, work, &lwork, &info);
  if( info > 0 ) {
    printf( "The algorithm failed to compute eigenvalues.\n" );
    exit( 1 );
  }  
  for(i = 0; i < dim; i++)
  {
    W[i] = W1[i];
  }
  free( (void*)work );
  free((void *) W1);

}

/*
 * mat_tran : transpose a given matrix
 *
 * Parameters : 
 * [In] m : int
 *        the column number of the original matrix
 * [In] n : int
 *        the row number of the original matrix
 * [In] a : double array
 *        the original matrix with shape (n, m)
 * [In] b : double array
 *        the array after transpose with shape (m, n)
*/
void mat_tran(int m, int n, double *a, double *b)
{
  int i, j;
  for(i = 0; i < m; i++)
  {
    for(j = 0; j < n; j++)
    {
      b[j + i * n] = a[i + j * m];
    }
  }
}