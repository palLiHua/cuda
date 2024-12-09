#include "../include/util.h"
#include <cuda.h>
#include <curand_kernel.h>

__global__ void initialize_state(curandState *states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(9384, tid, 0, states + tid);
}

__global__ void refill_randoms(float *dRand, int N, curandState *states)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;
    curandState *state = states + tid;

    for (i = tid; i < N; i += nthreads)
    {
        dRand[i] = curand_uniform(state);
    }
}

float cuda_device_rand()
{
    static cudaStream_t stream = 0;
    static curandState *states = NULL;
    static float *dRand = NULL;
    static float *hRand = NULL;
    static int dRand_length = 1000000;
    static int dRand_used = dRand_length;

    int threads_per_block = 256;
    int blocks_per_grid = 30;

    if (dRand == NULL)
    {
        CHECK(cudaStreamCreate(&stream));
        CHECK(cudaMalloc((void **)&dRand, sizeof(float) * dRand_length));
        CHECK(cudaMalloc((void **)&states, sizeof(curandState) *
                        threads_per_block * blocks_per_grid));
        hRand = (float *)malloc(sizeof(float) * dRand_length);
        initialize_state<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            states);
        refill_randoms<<<blocks_per_grid, threads_per_block>>>(dRand,
                dRand_length, states);  // 使用默认流，所以要等initialize_state执行完后启用
    }

    if (dRand_used == dRand_length)
    {
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(hRand, dRand, sizeof(float) * dRand_length,
                    cudaMemcpyDeviceToHost));
        refill_randoms<<<blocks_per_grid, threads_per_block, 0, stream>>>(dRand,
                dRand_length, states);
        dRand_used = 0;
    }

    return hRand[dRand_used++];
}

float cuda_host_rand()
{
    static cudaStream_t stream = 0;
    static float *dRand = NULL;
    static float *hRand = NULL;
    curandGenerator_t randGen;
    static int dRand_length = 1000000;
    static int dRand_used = 1000000;

    if (dRand == NULL)
    {
        CHECK_CURAND(curandCreateGenerator(&randGen,
                                           CURAND_RNG_PSEUDO_DEFAULT));
        CHECK(cudaStreamCreate(&stream));
        CHECK_CURAND(curandSetStream(randGen, stream));

        CHECK(cudaMalloc((void **)&dRand, sizeof(float) * dRand_length));
        hRand = (float *)malloc(sizeof(float) * dRand_length);
        CHECK_CURAND(curandGenerateUniform(randGen, dRand, dRand_length));
    }

    if (dRand_used == dRand_length)
    {
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(hRand, dRand, sizeof(float) * dRand_length,
                        cudaMemcpyDeviceToHost));
        CHECK_CURAND(curandGenerateUniform(randGen, dRand, dRand_length));
        dRand_used = 0;
    }

    return hRand[dRand_used++];
}

float host_rand()
{
    return (float)rand() / (float)RAND_MAX;
}

int main(int argc, char **argv)
{
    int i;
    int N = 8388608;

    for (i = 0; i < N; i++)
    {
        float h = host_rand();
        float d = cuda_host_rand();
        float dd = cuda_device_rand();
        printf("%2.4f %2.4f %2.4f\n", h, d, dd);
        getchar();
    }

    return 0;
}
