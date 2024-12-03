#include "../include/util.h"

#define TO_STRING(x) #x
#define STRINGIFY(x) TO_STRING(x)

#define TYPE double
#define N 10000

__global__ void warm_up(float *in, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        in[tid] += 1;
    }
}

__global__ void standard_kernel(TYPE a, TYPE *out, size_t iters)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid == 0)
    {
        TYPE tmp;

        for (size_t i = 0; i < iters; i++)
        {
            tmp = powf(a, 2.0f);
        }

        *out = tmp;
    }
}

__global__ void intrinsic_kernel(TYPE a, TYPE *out, size_t iters)
{
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if(tid == 0)
    {
        TYPE tmp;

        for (size_t i = 0; i < iters; i++)
        {
            tmp = __powf(a, 2.0f);
        }

        *out = tmp;
    }
}

int main(int argc, char **argv)
{
    int i;
    int runs = 1 << 10;
    size_t iters = 1<< 30;

    // warm-up
    size_t nBytes = sizeof(float) * N;
    float *h_tmp = (float*)malloc(nBytes);
    float *warm_up_i = NULL;
    CHECK(cudaMalloc((float**)&warm_up_i, nBytes));
    for (size_t i = 0; i < N; i++)
    {
        h_tmp[i] = (float)i;
    }
    CHECK(cudaMemcpy(warm_up_i, h_tmp, nBytes, cudaMemcpyHostToDevice))
    dim3 block(32);
    dim3 grid((N - 1) / 32 + 1);
    warm_up<<<grid,block>>>(warm_up_i, N);
    CHECK(cudaMemcpy(h_tmp, warm_up_i,  nBytes, cudaMemcpyDeviceToHost));
    printf("Warm-up finished\n");
    printf("TYPE = %s\n", STRINGIFY(TYPE));

    TYPE *d_standard_out, h_standard_out;
    CHECK(cudaMalloc((TYPE **)&d_standard_out, sizeof(TYPE)));

    TYPE *d_intrinsic_out, h_intrinsic_out;
    CHECK(cudaMalloc((TYPE **)&d_intrinsic_out, sizeof(TYPE)));

    // TYPE input_value = 8181.25;
    TYPE input_value = 8181.25;

    double mean_intrinsic_time = 0.0;
    double mean_standard_time = 0.0;

    for (i = 0; i < runs; i++)
    {
        double start_standard = cpuSecond();
        standard_kernel<<<1, 32>>>(input_value, d_standard_out, iters);
        CHECK(cudaDeviceSynchronize());
        mean_standard_time += cpuSecond() - start_standard;

        double start_intrinsic = cpuSecond();
        intrinsic_kernel<<<1, 32>>>(input_value, d_intrinsic_out, iters);
        CHECK(cudaDeviceSynchronize());
        mean_intrinsic_time += cpuSecond() - start_intrinsic;
    }

    CHECK(cudaMemcpy(&h_standard_out, d_standard_out, sizeof(TYPE),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_intrinsic_out, d_intrinsic_out, sizeof(TYPE),
                     cudaMemcpyDeviceToHost));
    TYPE host_value = powf(input_value, 2.0f);

    printf("Host calculated\t\t\t%f\n", host_value);
    printf("Standard Device calculated\t%f\n", h_standard_out);
    printf("Intrinsic Device calculated\t%f\n", h_intrinsic_out);
    printf("Host equals Standard?\t\t%s diff=%e\n",
           host_value == h_standard_out ? "Yes" : "No",
           fabs(host_value - h_standard_out));
    printf("Host equals Intrinsic?\t\t%s diff=%e\n",
           host_value == h_intrinsic_out ? "Yes" : "No",
           fabs(host_value - h_intrinsic_out));
    printf("Standard equals Intrinsic?\t%s diff=%e\n",
           h_standard_out == h_intrinsic_out ? "Yes" : "No",
           fabs(h_standard_out - h_intrinsic_out));
    printf("\n");
    printf("Mean execution time for standard function powf:    %f s\n",
           mean_standard_time);
    printf("Mean execution time for intrinsic function __powf: %f s\n",
           mean_intrinsic_time);

    return 0;
}
