#include <cuComplex.h>
#ifndef KERNEL_H
#define KERNEL_H

void matrixLaunch(cuDoubleComplex *d_A, cuDoubleComplex *d_B, cuDoubleComplex *d_C, int N);
void transposeLaunch(cuDoubleComplex *d_Q, cuDoubleComplex *d_P, int N);

#endif
