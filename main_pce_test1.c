/*
 * Numerical Example 1: 
 * --------------------
 * stochastic bottom and deterministic water surface.
 * 
 * Computational domain: [-1, 1]
 * 
 * g = 1.
 * 
 * theta = 1.3. 
 * 
 * Tlast = 0.8.
 * 
 * (alph, beta) = (0.0, 0.0), i.e., Legendre polynomials.
 * 
 * w = 1,   if x < 0,  
 *   = 0.5, if x > 0.
 * 
 * q = 0.
 * 
 * B = 0.125 * (cos(5 * pi * x) + 2.0) + 0.125 * \xi, if abs(x) < 0.2,
 *   = 0.125 + 0.125 * \xi ,                          otherwise.
 * 
 * the PCE of q is not filtering in this experiment.
 * 
 * Note: the state variables is stored in a double array with size (nx + 2) *
 *       2 * K_pce, where nx is the number of the grids, and K_pce is the order 
 *       of the PCE. In the i th row, 
 *            u[. + i * 2 * K_pce] = [\hat{w_i}, \hat{q_i}],
 *       i.e., the PCE of the ith cell averages of the water surface and the 
 *       discharges.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h>

#include "energy-stable-1d.h"
#include "opoly1d.h"

double B(double x, double xi);
double w(double x, double xi);
double q(double x, double xi);

int main(int argc, char *argv[])
{
  int i, j, k;
  int nx = 800, K_pce = 9; // spatial points and number of expansion
  int num_points = (int)(3 * (K_pce - 1) / 2 + 1); // number of quadrature points
  
  double *u, *bE, *bW, *bb; // state vector, east-side and west-side reconstructions of the bottom, cell average of the bottom, respectively.
 
  double *a, *b; // TTR coefficients
  int triples[2] = {0, 0};
  double *eval_points, *points, *weights, *trip_prod, normalizer;
  double xl, xr, x[nx+2], dx;
  double grav;
  double Tlast, t, dt;
  bool filter_q;
  FILE *fp1, *fp2, *fp3;
  clock_t start, end;
  double runtime;  
  char fname1[100], fname2[100], fname3[100];

  double alph = 0.0, bet = 0.0;
  int dzf = 125;
  start = clock();
  /* Allocation of memory */

  a = malloc(3 * K_pce * sizeof(double));
  b = malloc(3 * K_pce * sizeof(double));
  u = malloc((nx + 2) * 2 * K_pce * sizeof(double));
  bE = malloc((nx + 2) * K_pce * sizeof(double));
  bW = malloc((nx + 2) * K_pce * sizeof(double));
  bb = malloc((nx + 2) * K_pce * sizeof(double));
  weights = malloc(num_points * sizeof(double));
  points = malloc(num_points * sizeof(double));
  eval_points = malloc(num_points * K_pce * sizeof(double));  
  trip_prod = malloc(K_pce * K_pce * K_pce * sizeof(double));

  /* File creation */
  
  sprintf(fname1,"dz%d_N%d_nx%dstate_ex1-es1.txt",dzf, K_pce, nx); 
  sprintf(fname2,"dz%d_N%d_nx%db_ex1-es1.txt",dzf, K_pce, nx); 
  sprintf(fname3,"dz%d_N%d_nx%dmisc_ex1-es1.txt",dzf, K_pce, nx);  
  fp1 = fopen(fname1,"w");
  fp2 = fopen(fname2,"w");
  fp3 = fopen(fname3,"w");

  /* Orthonormal polynomials quantities to be used */

  jacobi_recurrence(3 * K_pce, alph, bet, a, b);
  Gaussian_quadrature1D(a, b, num_points, weights, points);
  opoly1d_eval(a, b, num_points, points, K_pce, eval_points);

  normalizer = tuple_product(a, b, K_pce, triples, 2, 0);
  
  for(i = 0; i < K_pce; i++)
  {
    for(j = 0; j < K_pce; j++)
    {
      triples[0] = i;
      triples[1] = j;
      for(k = 0; k < K_pce; k++)
      {
        trip_prod[k + j * K_pce + i * K_pce * K_pce] = \
          tuple_product(a, b, K_pce, triples, 2, k) / normalizer;
      }
    }
  }

  /* Constants */
  grav = 1.0;
  filter_q = false;

  /* Spatial grids */

  xl = -1.0;
  xr = 1.0;
  dx = (xr - xl) / nx;
  for(i = 0; i < nx + 2; i++)
  {
    x[i] = xl + (i - 0.5) * dx; 
  }

  /* Bottom topography */

  for(i = 0; i < nx + 2; i++)
  {
    for(j = 0; j < K_pce; j++)
    {
      bE[j + i * K_pce] = 0.0;
      bW[j + i * K_pce] = 0.0;
      for(k = 0; k < num_points; k++)
      {
        bE[j + i * K_pce] += weights[k] * eval_points[k + j * num_points] *\
          B(x[i] + 0.5 * dx, points[k]);
        bW[j + i * K_pce] += weights[k] * eval_points[k + j * num_points] *\
          B(x[i] - 0.5 * dx, points[k]);          
      }
      bb[j + i * K_pce] = 0.5 * (bE[j + i * K_pce] + bW[j + i * K_pce]);
    }
  }

  /* Water surface and discharges */

  for(i = 1; i < nx + 1; i++)
  {
    if(x[i] > 0)
    {
      u[i * 2 * K_pce] = 0.5;
    }
    else
    {
      u[i * 2 * K_pce] = 1.0;
    }
    u[K_pce + i * 2 * K_pce] = 0.0;

    for(j = 1; j < K_pce; j++)
    {      
      u[j + i * 2 * K_pce] = 0.0;
      u[(j + K_pce) + i * 2 * K_pce] = 0.0;
    }
  }

  /* Time and evolution */

  Tlast = 0.8;
  t = 0.0;
  while(t < Tlast)
  {
    /*
    evol3stage1d(bE, bW, trip_prod, dx, nx, K_pce, grav, th, eval_points, num_points,\
      filter_q, t, Tlast, u, &dt, fp3);    
    */
    //*
    //evol3stage1d_conv(bE, bW, trip_prod, dx, nx, K_pce, grav, th, eval_points, num_points,\
      filter_q, t, Tlast, u, &dt, fp3);
    //*/
    evol2stage1d_es1(bb, trip_prod, dx, nx, K_pce, grav, th, \
      eval_points, num_points, filter_q, t, Tlast, u, &dt, fp3);
    、、evol2stage1d_es2(bb, bE, bW, trip_prod, dx, nx, K_pce, grav, \
     eval_points, num_points, filter_q, t, Tlast, u, &dt, fp3);       
    t += dt;
    printf("%.16f\n", t);
  }

  /* Output data */

  for(i=1; i<=nx; i++)
  {
    // Output resulting PCE coefficients
    fprintf(fp1,"%.16f\t",x[i]);
    fprintf(fp2,"%.16f\t",x[i]);
    for(j = 0; j < 2 * K_pce; j++)
    {
      fprintf(fp1,"%.16f\t", u[j + i * 2 * K_pce]);      
    }
    for(j = 0; j < K_pce; j++)
    {
      fprintf(fp2,"%.16f\t", bb[j + i * K_pce]);
    }
    fprintf(fp1,"\n");
    fprintf(fp2,"\n");		
  }

  /* Free space */

  free((void *) u); free((void *) bb); 
  free((void *) bE); free((void *) bW);
  free((void *) a); free((void *) b); 
  free((void *) eval_points); free((void *) weights);
  free(trip_prod);

  /* Close file*/
  fclose(fp1); fclose(fp2); fclose(fp3);
  end = clock();
  runtime = ((double)(end-start))/(CLOCKS_PER_SEC);
  printf("The runtime the program is %.16f\n", runtime);  
  return 0;
}

/*
 * Function: B
 * -----------
 * compute the pointwise evaluated uncertain bottom
 * 
 * [In]  x: the location of the point.
 * 
 * [In]  xi: the stochastic parameter of the bottom.
 * 
 * [Out] result: the evaluated bottom.  
 */

double B(double x, double xi)
{
	double pi = 3.14159265358979323;
	double result;
	if((x < 0.2 + 1e-16)&&(x > -0.2 - 1e-16))
		{result = 0.125 * (cos(5 * pi * x) + 2.0);}// + 0.1 * xi;
	else
		{result = 0.125;}// + 0.1 * xi;}
  //result = 0;
	return result;
}

/*
 * Function: w
 * -----------
 * compute the pointwise evaluated uncertain water surface
 * 
 * [In]  x: the location of the point.
 * 
 * [In]  xi: the stochastic parameter of the surface.
 * 
 * [Out] result: the evaluated surface.  
 */

double w(double x, double xi)
{
  double result;

  if(x > 0)
  {
    result = 0.5;
  }
  else 
  {
    result = 1.0;
  }

  return result;  
}

/*
 * Function: q
 * -----------
 * compute the pointwise evaluated uncertain discharge
 * 
 * [In]  x: the location of the point.
 * 
 * [In]  xi: the stochastic parameter of the discharge.
 * 
 * [Out] result: the evaluated discharge.  
 */

double q(double x, double xi)
{
  double result = 0.0;
  
  return result;
}
