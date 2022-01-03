#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>


void jacobi_recurrence(int K_pce, double alpha, double beta, double *a, \
	double *b);
double tuple_product(double *a, double *b, int K_pce, int *n, int nl, int k);
void Gaussian_quadrature1D(double *a, double *b, int num_points, \
	double *weights, double *points);
void opoly1d_eval(double *a, double *b, int num_points, double *points, \
	double K_pce, double *eval);
void QsortI(int *n, int low, int high);
int partition(int n[], int low, int high);
extern void dsyev_(char*, char*, int*, double*, int*, double*, double*, \
	int*, int*);

/*
 * Function: jacobi_recurrence
 * ---------------------------
 * compute the first K_pce three term recurrence coefficients for the Jacobi
 * polynomials with parameter (alpha, beta).
 * 
 * [In] 		K_pce: number of recurrence coefficients. In order to calculate 
 * 								 the triple product, K_pce should be set to three times 
 * 								 the number of polynomials used in the PCE.
 * 
 * [In] 		alpha: parameter
 * 
 * [In] 		beta: parameter
 * 
 * [In/Out] a: the coeffcients {a_j}
 * 
 * [In/Out] b: the coeffcients {b_j}
*/

void jacobi_recurrence(int K_pce, double alpha, double beta, double *a, \
 double *b)
{
	double tmp = (beta * beta - alpha * alpha);
	int j;
	assert((alpha>-1) && (beta > -1));

	if(K_pce > 0)
	{
		a[0] = (beta - alpha) / (alpha + beta + 2.0);
		b[0] = exp((alpha + beta + 1.0) * log(2.0) + lgamma(alpha + 1.0) + \
		lgamma(beta + 1.0) - lgamma(alpha + beta + 2.0));
	}

	if (K_pce > 1)
	{
		a[1] = tmp / ((2.0 + alpha + beta) * (4.0 + alpha + beta));
		b[1] = 4 * (1.0 + alpha) * (1.0 + beta) / (pow(2.0 + alpha + beta, 2.0)*\
				(3.0 + alpha + beta));			
	}

	for(j = 2; j < K_pce; j++)
	{
		a[j] = tmp / ((2.0 * j + alpha + beta)*\
				(2.0 * j + alpha + beta + 2.0));
		b[j] = 4 * j * (j + alpha) * (j + beta) * (j + alpha + beta);
		b[j] /= (pow(2.0 * j + alpha + beta, 2.0) * \
				(2.0 * j + alpha + beta + 1.0) * (2 * j + alpha + beta - 1.0));		
	}

	for(j = 0; j < K_pce; j++)
		b[j] = sqrt(b[j]);
}

/*
 * Function: tuple_product
 * -----------------------
 * compute the tuple product <p_{n_1}p_{n_2}...p_{n_K_pce}, p_k>
 * provided the three term recurrence relation coefficients of the orthornormal
 * polynomials, where n_1, n_2, ..., n_K_pce are the indexes of the polynomials.
 * 
 * [In] a: {a_{n+1}} in TTR
 * 
 * [In] b: {b_n} in TTR
 * 
 * [In] K_pce: the highest degree in the orthornormal polynomial expansion.
 * 
 * [In] n: the index of the first (K_pce - 1) orthonormal polynomials.
 * 
 * [In] nl: length of the vector n.
 * 
 * [In] k: the index of the last orthonormal polynomial in the tuple product.
 */

double tuple_product(double *a, double *b, int K_pce, int *n, int nl, int k)
{
	int i, M, sum1 = 0;
	double v[(nl + 1) * K_pce + 1], w[(nl + 1) * K_pce + 1];
	double vtmp[(nl + 1) * K_pce + 1], Jv[(nl + 1) * K_pce + 1];
	int nq, jq;

	if (nl == 0)
	{
		if (k == 0) return b[0];
		return 0;
	}

	QsortI(n, 0, nl - 1);
	
	for (i = 0; i < nl; i++)
	{
		sum1 += n[i];
	}

	M = sum1 + k;

	for (i = 0; i <= M; i++)
		v[i] = 0.0;
	v[n[0]] = 1.0;

	for (nq = 1; nq < nl; nq++)
	{
		for (i = 0; i <= M; i++)
		{
			v[i] = v[i] / b[0];
			w[i] = 0.0;
		}
		for (jq = 0; jq < n[nq]; jq++)
		{
			Jv[0] = a[0] * v[0] + b[1] * v[1];
			for (i = 1; i <= M; i++)    
			{
				Jv[i] = a[i] * v[i] + b[i + 1] * v[i + 1] + b[i] * v[i - 1];
			}  
			Jv[M] = a[M] * v[M] + b[M] * v[M - 1];

			for (i = 0; i <= M; i++)
			{
				vtmp[i] = v[i];
				v[i] = (-a[jq] * v[i] - b[jq] * w[i] + Jv[i]) / b[jq + 1];
				w[i] = vtmp[i];               
			}       
		}
	}

	return v[k];
}

/*
 * Function: Gaussian_quadrature1D
 * -------------------------------
 * compute the nodes and corresponding weights of the Gaussian quadrature rule, 
 * provided three-term recurrence coefficients of the Jacobi polynomial family
 * (on [-1, 1]). The length of the input coefficients len(a), len(b) should be 
 * greater than the number of quadrature points num_points.
 * 
 * The three term recurrence relation:
 * 	b_{n + 1}p_{n + 1}(x) = (x - a_{n})p_n(x) - b_{n}p_n(x).
 * 
 * [In]  		a: the {a_n} in the three-term recurrence relation.
 * 
 * [In]  		b: the {b_n} in the three-term recurrence relation.
 * 				
 * [In]	 		num_points: the number of the quadrature points.
 * 
 * [In/Out] weights: vector of weights
 * 
 * [In/Out] points: quadrature points
 */

void Gaussian_quadrature1D(double *a, double *b, int num_points, \
	double *weights, double *points)
{
	int i, j;
	int lda = num_points, info, lwork;
	double w[num_points], *work, wkopt;
	double *J;
	J = malloc(num_points * num_points * sizeof(double));

	/* Calculating Gaussian quadrature points */
	for(i = 0;i < num_points; i++)
		for(j = 0;j < num_points; j++)
			J[j + i * num_points] = 0.0;
	
	for(i = 0; i < num_points; i++)
	{
		J[i + i * num_points] = a[i];
	}
	for(i = 0; i < num_points - 1; i++)
	{
		J[(i + 1) + i * num_points] = b[i + 1];
		J[i + (i + 1) * num_points] = b[i + 1];
	}
	lwork = -1;
	dsyev_( "V", "U", &num_points, J, &lda, w, &wkopt, &lwork, &info );
	lwork = (int)wkopt;
  work = (double*)malloc(lwork*sizeof(double));
	dsyev_( "V", "U", &num_points, J, &lda, w, work, &lwork, &info );
	for(i=0; i<num_points;i++)
	{
		points[i] = w[i];
		weights[i] = J[i * num_points] * J[i * num_points];
	}
	
	free(work);	
}

/*
 * Function: opoly1d_eval
 * ----------------------------------
 * evaluate the orthonormal polynomials with given TTR coefficients
 * at given points.
 * 
 * [In]  		a: {a_{n+1}} in TTR coefficients.
 * 
 * [In]  		b: {b_n} in TTR coefficients.
 * 
 * [In]  		num_points: number of points
 * 
 * [In]  		points: the vector of points {\xi_k}_{k=1}^{num_points}
 * 
 * [In]  		K_pce: the highest degree of the polynomial
 * 
 * [In/Out] eval: the orthonormal polynomials evaluated at given points,
 * 					eval[k + j * num_points] = p_{j}(\xi_k)
 */

void opoly1d_eval(double *a, double *b, int num_points, double *points, \
	double K_pce, double *eval)
{
	int i, j;
	for(i = 0; i < num_points; i++)
		eval[i] = 1.0;
	if(K_pce > 0)
	{
		for(i = 0; i < num_points; i++)
			eval[i + num_points] = 1.0 / b[1] * ((points[i] - a[0]) * eval[i]);
	}
	for(j = 2; j < K_pce; j++)
		for(i = 0; i < num_points; i++)
			eval[i + j * num_points] = 1.0 / b[j]* ((points[i] - a[j - 1]) * \
			eval[i + (j - 1) * num_points] - b[j - 1] * eval[i + (j - 2) * num_points]);
}

/*
 * Function: QsortI
 * ----------------
 * sort the given array in descending order.
 * 
 * [In/Out] arr: input array. On output, the input array is sorted in 
 * 							 descending order.
 * 
 * [In]			low: the starting position.
 * 
 * [In]			high: the ending position
 */

void QsortI(int arr[], int low, int high)
{
	int pi;
	if (low < high)
	{
		/* The partitioning position. arr[pi] is at the correct position*/
		pi = partition(arr, low, high); 

		/* Divide and conquer */
		QsortI(arr, low, pi - 1);
		QsortI(arr, pi + 1, high);
	}
}

/*
 * Function: partition
 * -------------------
 * this function takes the last element in the subarray as a pivot.
 * On output, the last element is in its correct position in the subarray. 
 * The elements that are greater than the pivot is on the left of the pivot, 
 * and the elements that are less than the pivot is on the right of the pivot.
 * 
 * [In/Out] arr: input array. On output, the pivot is on the correct position 
 * 							 of the array.
 * 
 * [In]     low: starting position.
 * 
 * [In]     high: end position (pivot position).
 * 
 * [Out]		return the correct position of the pivot.
 */

int partition(int arr[], int low, int high)
{
	int j, tmp;
	int i = low - 1;
	for (j = low; j <= high - 1; j++)
	{
		if (arr[j] > arr[high])
		{
			i++;
			tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
		}
	}
	tmp = arr[i + 1];
	arr[i + 1] = arr[high];
	arr[high] = tmp;   
	return (i + 1);
}
