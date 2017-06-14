/* -*- Mode:C; Coding:us-ascii-unix; fill-column:132 -*- */
/* ****************************************************************************************************************************** */
/**
   @file      blas3C.c
   @author    Mitch Richling <https://www.mitchr.me/>
   @Copyright Copyright 1997 by Mitch Richling.  All rights reserved.
   @brief     Demonstrate several cblas (level 1) functions. @EOL
   @Keywords  blas cblas C fortran numerical linear algebra vector matrix gemv ger
   @Std       C89

   This is a simple program intended to illistrate how to make use of #gemv and #ger blas routines (as implimented in the cblas).

*/

/* ------------------------------------------------------------------------------------------------------------------------------ */

#include <stdio.h>              /* I/O lib         ISOC     */
#include <stdlib.h>             /* Standard Lib    ISOC     */
#include <time.h>
#include "blaio.h"              /* Basic Linear Algebra I/O */
#include "omp.h"

int main(int argc, char **argv) {
	// Info OpenMP
        int tid;
        int nthreads=1;
	if (nthreads>1)
//		printf("id\t#OpenMP\tCPU\n");
	#pragma omp parallel shared(nthreads)
	{
	        nthreads = omp_get_num_threads();
	        #pragma omp critical
	        {
		        tid = omp_get_thread_num();
//	                printf("%d\t%d\t%d\n", tid, nthreads, sched_getcpu());
	        }
	}

	int size=atoi(argv[1]);
	double a[size*size];
	double b[size*size];
	double c[size*size];
	int i;
	for(i=0; i<size*size; ++i) {
		a[i]=i;
		b[i]=i*2;
		c[i]=i;
	}

	clock_t begin = clock();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size,    size,    size, 1.0,   a,   size, b, size,  0.0, c,  size);
	clock_t end = clock();

	/*
	- Count Total Operations (C <- alpha * AB + beta*C):
	Operations(AB) = MNK (mult) + MN(K-1) (add)
	Operation(alpha*AB) = MNK (mult) + MN(K-1) (add) + MN (mult) 
	Operation(alpha*AB + beta*C) = MNK (mult) + MN(K-1) (add) + MN (mult) + MN (mult) + MN (add)
	Total = MN(K+2) (mult) + MNK (add)

	- GFLOPS for DGEMM and SGEMM:
	Total Operations = MN(2K+2) GLOPS = (MN(2K+2) / (1000^3 * (timeInSec))
	*/

	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
//	printf("Total CPU time: %f\n", time_spent);
//	printf("Execution time: %f\n", time_spent/nthreads);
	int new_size=size/1000;
	double gflops=1.0*new_size*new_size*(2*size+2)/(1000.0*time_spent);
//	printf("Gflops: %f\n", gflops);
	printf("%f", gflops);

//	printMatrix(CblasRowMajor, size, size, a, 8, 3, NULL, NULL, NULL, NULL, NULL, "c <- 1.0*a*b+0.0*c = ");
//	printMatrix(CblasRowMajor, size, size, b, 8, 3, NULL, NULL, NULL, NULL, NULL, "c <- 1.0*a*b+0.0*c = ");
//	printMatrix(CblasRowMajor, size, size, c, 8, 3, NULL, NULL, NULL, NULL, NULL, "c <- 1.0*a*b+0.0*c = ");

	return 0;
}
