#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <stdio.h>
#include "kernel.h"
#define N 1000


__global__
int main()
{
	cuDoubleComplex *A = 0;
	cuDoubleComplex *B = 0;
	cuDoubleComplex *C = 0;
	cuDoubleComplex *Q = 0;
	cuDoubleComplex *P = 0;

	cudaMallocManaged(&A, N*N * sizeof(cuDoubleComplex));
	cudaMallocManaged(&B, N*N * sizeof(cuDoubleComplex));
	cudaMallocManaged(&C, N*N * sizeof(cuDoubleComplex));
	cudaMallocManaged(&Q, N*N * sizeof(cuDoubleComplex));
	cudaMallocManaged(&P, N*N * sizeof(cuDoubleComplex));

	transposeLaunch(Q, P, N);

	matrixLaunch(P,A,C,N);
	matrixLaunch(C, Q, B, N);
}