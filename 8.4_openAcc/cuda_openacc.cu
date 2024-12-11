#include "../include/util.h"
#include <curand_kernel.h>
#include <cublas_v2.h>

#define M 1024
#define N 1024
#define P 1024

int main(int argc, char **argv)
{
    int i,j,k;
    float *__restrict__ d_A;    // Same as float *restrict d_A
    float *__restrict__ d_B;
    float *__restrict__ d_C;
    
    float *d_row_sums;
    float total_sum;

    curandGenerator_t rand_state = 0;
    cublasHandle_t cublas_handle = 0;

    CHECK_CURAND(curandCreateGenerator(&rand_state, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    CHECK(cudaMalloc((void **)&d_A, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&d_B, sizeof(float) * N * P));
    CHECK(cudaMalloc((void **)&d_C, sizeof(float) * M * P));
    
    CHECK(cudaMalloc((void **)&d_row_sums, sizeof(float) * M));

    CHECK_CURAND(curandGenerateUniform(rand_state, d_A, M * N));
    CHECK_CURAND(curandGenerateUniform(rand_state, d_B, N * P));

#pragma acc parallel loop gang deviceptr(d_A, d_B, d_C)
    
    for (i = 0; i < M; i++)
    {
#pragma acc loop worker vector

        for (j = 0; j < P; j++)
        {
            float sum = 0.0f;
            for (k = 0; k < N; k++)
            {
                sum += d_A[i * N + k] * d_B[k * P + j];
            }
            d_C[i * P + j] = sum;
        }
    }

    CHECK_CUBLAS(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

    for (i = 0; i < M; i++)
    {
        CHECK_CUBLAS(cublasSasum(cublas_handle, P, d_C + (i * P), 1, d_row_sums + i));
    }

    CHECK_CUBLAS(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));

    CHECK_CUBLAS(cublasSasum(cublas_handle, M, d_row_sums, 1, &total_sum));
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_row_sums));

    printf("Total sum = %f\n", total_sum);

    return 0;
}