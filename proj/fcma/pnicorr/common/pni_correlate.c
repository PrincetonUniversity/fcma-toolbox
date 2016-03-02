//
//  pni_correlate.c
//  pni_correlation
//
//  Created by Ben Singer on 1/19/16.
//  Copyright © 2016 Ben Singer. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mkl.h>
#include <math.h>

// Starting point: online PDF "Benchmarking the Intel® Xeon PhiTM Coprocessor"
// [ Infrared Processing and Analysis Center, Caltech, F. Masci, 09/04/2013 ]

/*	Program pni_correlate: multiply two matrices using a highly thread-optimized 
	routine from the Intel Math Kernel Library (MKL). */
/*----------------------------------------------------------------------*/

/* scaling factors needed for sgemm() */
static const float a = 1.0f;
static const float b = 0.0f;

/* Multiply matrix A by B and put result in C. All should be size nxn */
/* No scaling, transposing, in-place accumulation etc etc */
#define SGEMM_SIMPLE_SQUARE(A,B,C,n) \
	sgemm("N","N",&n,&n,&n,&a,A,&n,B,&n,&b,C,&n)
	
/* Multiply two square matrices A and B to generate matrix C using the
optimized sgemm() routine (for single precision floating point) from MKL. */
float fastMult(int size,
               float (* restrict A)[size],
               float (* restrict B)[size],
               float (* restrict C)[size],
               int nIter)
{
	float (*restrict At)[size] = malloc(sizeof(float)*size*size);
	float (*restrict Bt)[size] = malloc(sizeof(float)*size*size);
	float (*restrict Ct)[size] = malloc(sizeof(float)*size*size);
	
	/* transpose input matrices to get better sgemm() performance. */

    // NOTE this assumes input matrices were not already transposed,
    // putting burden on input might be better than assuming anything here

  	#pragma omp parallel for
	for(int i=0; i < size; i++) {
		for(int j=0; j < size; j++) {
			At[i][j] = A[j][i];
			Bt[i][j] = B[j][i];
		}
	}

	/* warm up run to overcome setup overhead in benchmark runs below. */
	SGEMM_SIMPLE_SQUARE(At,Bt,Ct,size);

	double StartTime=dsecnd();
	for(int i=0; i < nIter; i++) {
		SGEMM_SIMPLE_SQUARE(At,Bt,Ct,size);
	}
	double EndTime=dsecnd();
	
	float tottime = EndTime - StartTime;
	float avgtime = tottime / nIter;
	printf("tot runtime = %f sec\n", tottime);
	printf("avg runtime per vec. mult. = %f sec\n", avgtime);
	float GFlops = (2e-9*size*size*size)/avgtime;

	free(At);
	free(Bt);
	free(Ct);
	
	return ( GFlops );
}

/*----------------------------------------------------------------------*/
/* Read input parameters; set-up dummy input data; multiply matrices using
the fastMult() function above; average repeated runs therein. */
int main(int argc, char *argv[])
{
	if(argc != 4) {
		fprintf(stderr,"Use: %s size nThreads nIter\n",argv[0]); return -1;
	}
	int i,j,nt;
	
	int size=atoi(argv[1]);
	int nThreads=atoi(argv[2]);
	int nIter=atoi(argv[3]);
	
	omp_set_num_threads(nThreads);
	mkl_set_num_threads(nThreads);
	
	/* when compiled in "mic-offload" mode, this memory gets allocated on host, 
	when compiled in "mic-native" mode, it gets allocated on mic. */
	float (*restrict A)[size] = malloc(sizeof(float)*size*size);
	float (*restrict B)[size] = malloc(sizeof(float)*size*size);
	float (*restrict C)[size] = malloc(sizeof(float)*size*size);
	
	/* this first pragma is just to get the actual #threads used (sanity check). */
	 #pragma omp parallel
	{
		nt = omp_get_num_threads();
		
		/* Fill the A and B arrays with dummy test data. */
		#pragma omp parallel for default(none) shared(A,B,size) private(i,j)
		for(i = 0; i < size; ++i) {
			for(j = 0; j < size; ++j) {
				A[i][j] = (float)i + j;
				B[i][j] = (float)i - j;
			}
		}
	}

	/* run the matrix multiplication function nIter times and average runs therein. */
	float Gflop = fastMult(size,A,B,C,nIter);
	printf("size = %d x %d; nThreads = %d; #GFlop/s = %g\n", size, size, nt, Gflop);

	free(A);
	free(B);
	free(C);
	
	return 0;
}
