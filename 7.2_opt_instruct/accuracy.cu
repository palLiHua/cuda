#include "../include/util.h"

__global__ void kernel(float *F, double *D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        *F = 12.1;
        *D = 12.1;
        printf("Device single-precision representation of 12.1 in Device = %.20f\n", *F);
        printf("Device double-precision representation of 12.1 in Device = %.20f\n", *D);
    }
}

int main(int argc, char **argv)
{
    float h_f = 12.1;   // host float
    double h_d = 12.1;  // host double

    printf("Host single-precision representation of 12.1   = %.20f\n", h_f);
    printf("Host double-precision representation of 12.1   = %.20f\n", h_d);
    
    float *h_f_ref = (float*)malloc(sizeof(float));     // copy back from gpu
    double *h_d_ref = (double*)malloc(sizeof(double));
    float *d_f; // device float
    double *d_d;

    CHECK(cudaMalloc((float **)&d_f, sizeof(float)));
    CHECK(cudaMalloc((double **)&d_d, sizeof(double)));

    dim3 block(1);
    dim3 grid(1);

    kernel<<<grid, block>>>(d_f, d_d);
    
    CHECK(cudaMemcpy(h_f_ref, d_f, sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_d_ref, d_d, sizeof(double),
                     cudaMemcpyDeviceToHost));


    printf("Device single-precision representation of 12.1 = %.20f\n", *h_f_ref);
    printf("Device double-precision representation of 12.1 = %.20f\n", *h_d_ref);
    printf("Device and host single-precision representation equal? %s\n",
           *h_f_ref == h_f ? "yes" : "no");
    printf("Device and host double-precision representation equal? %s\n",
           *h_d_ref == h_d ? "yes" : "no");

    cudaFree(d_f);
    cudaFree(d_d);

    free(h_f_ref);
    free(h_d_ref);

    return 0;
}

