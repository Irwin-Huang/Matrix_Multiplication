#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <stdio.h>
#include "kernel.h"
#define TX 32
#define TY 32


__global__
void matrixMult(cuDoubleComplex *d_A, cuDoubleComplex *d_B, cuDoubleComplex *d_C, int N)
{
	const int idx = threadIdx.x + blockDim.x*blockIdx.x;
	const int idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	if ((idx >= N) || (idy >= N)) return;

	cuDoubleComplex tempProd = make_cuDoubleComplex(0,0);
	for (int i = 0;i < N; i++) 
	{
		tempProd = cuCmul(d_A[N*idy + i], d_B[N*i + idx]);
	}

	d_C[N*idy+idx] = tempProd;
}

void matrixLaunch(cuDoubleComplex *d_A, cuDoubleComplex *d_B, cuDoubleComplex *d_C, int N)
{
	const dim3 blockSize(TX, TY);
	const dim3 gridSize((N + TX - 1) / TX, (N + TY - 1) / TY);
	matrixMult << <gridSize, blockSize >> > (d_A, d_B, d_C, N);

}

__global__
void transposeMat(cuDoubleComplex *d_Q, cuDoubleComplex *d_P, int N)
{
	const int idx = threadIdx.x + blockDim.x*blockIdx.x;
	const int idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	if ((idx >= N) || (idy >= N)) return;

	d_Q[N*idy + idx] = cuConj(d_P[N*idx + idy]);

}

void transposeLaunch(cuDoubleComplex *d_Q, cuDoubleComplex *d_P, int N)
{
	const dim3 blockSize(TX, TY);
	const dim3 gridSize((N + TX - 1) / TX, (N + TY - 1) / TY);
	transposeMat << <gridSize, blockSize >> > (d_Q, d_P, N);

}