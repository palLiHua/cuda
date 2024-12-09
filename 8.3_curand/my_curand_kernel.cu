#include "../include/util.h"
#include <cuda.h>
#include <curand_kernel.h>

int threads_per_block = 256;
int blocks_per_grid = 30;

__global__ void device_api_kernel(curandState *states, float *out, int N)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;
    curandState *state = states + tid;

    curand_init(9384, tid, 0, state);

    for (i = tid; i < N; i += nthreads)
    {
        float rand = curand_uniform(state);
        rand = rand * 2;
        out[i] = rand;
    }
}

__global__ void host_api_kernel(float *randomValues, float *out, int N)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;

    for (i = tid; i < N; i += nthreads)
    {
        float rand = randomValues[i];
        rand = rand * 2;
        out[i] = rand;
    }
}

void use_host_api(int N)
{
    curandGenerator_t randGen;
    CHECK_CURAND(curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT));

    float *dRand, *dOut, *hOut;
    CHECK(cudaMalloc((float**)&dRand, sizeof(float) * N));
    CHECK(cudaMalloc((float**)&dOut, sizeof(float) * N));

    hOut = (float*)malloc(sizeof(float) * N);

    CHECK_CURAND(curandGenerateUniform(randGen, dRand, N));

    host_api_kernel<<<blocks_per_grid, threads_per_block>>>(dRand, dOut, N);

    CHECK(cudaMemcpy(hOut, dOut, sizeof(float) * N, cudaMemcpyDeviceToHost));

    printf("Sampling of output from host API:\n");

    for (int i = 0; i < 10; i++)
    {
        printf("%2.4f\n", hOut[i]);
    }

    printf("...\n");

    free(hOut);
    CHECK(cudaFree(dRand));
    CHECK(cudaFree(dOut));
    CHECK_CURAND(curandDestroyGenerator(randGen));
}

void use_device_api(int N)
{
    static curandState *states = NULL;
    CHECK(cudaMalloc((void **)&states, sizeof(curandState) *
                threads_per_block * blocks_per_grid));
    
    float *dOut, *hOut;

    CHECK(cudaMalloc((void **)&dOut, sizeof(float) * N));
    hOut = (float *)malloc(sizeof(float) * N);

    device_api_kernel<<<blocks_per_grid, threads_per_block>>>(states, dOut, N);

    CHECK(cudaMemcpy(hOut, dOut, sizeof(float) * N, cudaMemcpyDeviceToHost));
    
    printf("Sampling of output from device API:\n");

    for (int i = 0; i < 10; i++)
    {
        printf("%2.4f\n", hOut[i]);
    }

    printf("...\n");

    free(hOut);
    CHECK(cudaFree(dOut));
    CHECK(cudaFree(states));

}

int main(int argc, char **argv)
{
    int N = 8388608;

    use_host_api(N);
    use_device_api(N);

    return 0;
}