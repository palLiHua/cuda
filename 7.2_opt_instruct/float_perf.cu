#include "../include/util.h"

__global__ void warm_up(float *in, size_t N)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        float tmp = in[tid] + 1;
    }
}
__global__ void lots_of_float_compute(float *inputs, int N, size_t niters,
                                      float *outputs)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // size_t nthreads = gridDim.x * blockDim.x;

    if (tid < N)
    {
        size_t iter;
        float val = inputs[tid];

        for (iter = 0; iter < niters; iter++)
        {
            val = (val + 5.0f) - 101.0f;
            val = (val / 3.0f) + 102.0f;
            val = (val + 1.07f) - 103.0f;
            val = (val / 1.037f) + 104.0f;
            val = (val + 3.00f) - 105.0f;
            val = (val / 0.22f) + 106.0f;
        }

        outputs[tid] = val;
    }
}


__global__ void lots_of_double_compute(double *inputs, int N, size_t niters,
                                       double *outputs)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // size_t nthreads = gridDim.x * blockDim.x;

    if (tid < N)
    {
        size_t iter;
        double val = inputs[tid];

        for (iter = 0; iter < niters; iter++)
        {
            val = (val + 5.0) - 101.0;
            val = (val / 3.0) + 102.0;
            val = (val + 1.07) - 103.0;
            val = (val / 1.037) + 104.0;
            val = (val + 3.00) - 105.0;
            val = (val / 0.22) + 106.0;
        }

        outputs[tid] = val;
    }
}


static void run_float_test(size_t N, int niters, int blocksPerGrid,
                           int threadsPerBlock, double *toDeviceTime,
                           double *kernelTime, double *fromDeviceTime,
                           float *sample, int sampleLength)
{
    int i;
    float *h_floatInputs, *h_floatOutputs;
    float *d_floatInputs, *d_floatOutputs;

    size_t nBytes = sizeof(float) * N;

    h_floatInputs = (float *)malloc(nBytes);
    h_floatOutputs = (float *)malloc(nBytes);
    CHECK(cudaMalloc((float **)&d_floatInputs, nBytes));
    CHECK(cudaMalloc((float **)&d_floatOutputs, nBytes));

    for (i = 0; i < N; i++)
    {
        h_floatInputs[i] = (float)i;
    }

    double toDeviceStart = cpuSecond();
    CHECK(cudaMemcpy(d_floatInputs, h_floatInputs, nBytes,
                     cudaMemcpyHostToDevice));
    *toDeviceTime = cpuSecond() - toDeviceStart;

    double kernelStart = cpuSecond();
    lots_of_float_compute<<<blocksPerGrid, threadsPerBlock>>>(d_floatInputs,
            N, niters, d_floatOutputs);
    CHECK(cudaDeviceSynchronize());
    *kernelTime = cpuSecond() - kernelStart;

    double fromDeviceStart = cpuSecond();
    CHECK(cudaMemcpy(h_floatOutputs, d_floatOutputs, nBytes,
                     cudaMemcpyDeviceToHost));
    *fromDeviceTime = cpuSecond() - fromDeviceStart;

    for (i = 0; i < sampleLength; i++)
    {
        sample[i] = h_floatOutputs[i];
    }

    CHECK(cudaFree(d_floatInputs));
    CHECK(cudaFree(d_floatOutputs));
    free(h_floatInputs);
    free(h_floatOutputs);
}


static void run_double_test(size_t N, int niters, int blocksPerGrid,
                            int threadsPerBlock, double *toDeviceTime,
                            double *kernelTime, double *fromDeviceTime,
                            double *sample, int sampleLength)
{
    int i;
    double *h_doubleInputs, *h_doubleOutputs;
    double *d_doubleInputs, *d_doubleOutputs;

    size_t nBytes = sizeof(double) * N;

    h_doubleInputs = (double *)malloc(nBytes);
    h_doubleOutputs = (double *)malloc(nBytes);
    CHECK(cudaMalloc((void **)&d_doubleInputs, nBytes));
    CHECK(cudaMalloc((void **)&d_doubleOutputs, nBytes));

    for (i = 0; i < N; i++)
    {
        h_doubleInputs[i] = (double)i;
    }

    double toDeviceStart = cpuSecond();
    CHECK(cudaMemcpy(d_doubleInputs, h_doubleInputs, nBytes,
                     cudaMemcpyHostToDevice));
    *toDeviceTime = cpuSecond() - toDeviceStart;

    double kernelStart = cpuSecond();
    lots_of_double_compute<<<blocksPerGrid, threadsPerBlock>>>(d_doubleInputs,
            N, niters, d_doubleOutputs);
    CHECK(cudaDeviceSynchronize());
    *kernelTime = cpuSecond() - kernelStart;

    double fromDeviceStart = cpuSecond();
    CHECK(cudaMemcpy(h_doubleOutputs, d_doubleOutputs, nBytes,
                     cudaMemcpyDeviceToHost));
    *fromDeviceTime = cpuSecond() - fromDeviceStart;

    for (i = 0; i < sampleLength; i++)
    {
        sample[i] = h_doubleOutputs[i];
    }

    CHECK(cudaFree(d_doubleInputs));
    CHECK(cudaFree(d_doubleOutputs));
    free(h_doubleInputs);
    free(h_doubleOutputs);
}

int main(int argc, char **argv)
{
    int i;
    double meanFloatToDeviceTime, meanFloatKernelTime, meanFloatFromDeviceTime;
    double meanDoubleToDeviceTime, meanDoubleKernelTime,
           meanDoubleFromDeviceTime;
    struct cudaDeviceProp deviceProperties;
    size_t totalMem, freeMem;
    float *floatSample;
    double *doubleSample;
    int sampleLength = 10;
    int nRuns = 1;
    int nKernelIters = 10;

    meanFloatToDeviceTime = meanFloatKernelTime = meanFloatFromDeviceTime = 0.0;
    meanDoubleToDeviceTime = meanDoubleKernelTime =
                                 meanDoubleFromDeviceTime = 0.0;

    CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

    size_t N = (freeMem * 0.02) / sizeof(double);    // Number of elements, choose small number if your gpu's mem is large
    int threadsPerBlock = 256;
    int blocksPerGrid = (N - 1) / threadsPerBlock + 1;

    if (blocksPerGrid > deviceProperties.maxGridSize[0])
    {
        blocksPerGrid = deviceProperties.maxGridSize[0];
    }

    printf("Running %d blocks with %d threads/block over %lu elements\n",
           blocksPerGrid, threadsPerBlock, N);

    
    
    size_t nBytes = sizeof(float) * N;
    float *h_tmp = (float*)malloc(nBytes);
    float *warm_up_i = NULL;
    CHECK(cudaMalloc((float**)&warm_up_i, nBytes));
    for (size_t i = 0; i < N; i++)
    {
        h_tmp[i] = (float)i;
    }
    CHECK(cudaMemcpy(warm_up_i, h_tmp, nBytes, cudaMemcpyHostToDevice))
    dim3 block(threadsPerBlock);
    dim3 grid(blocksPerGrid);
    warm_up<<<grid,block>>>(warm_up_i, N);
    CHECK(cudaMemcpy(h_tmp, warm_up_i,  nBytes, cudaMemcpyDeviceToHost));
    printf("Warm-up finished");

    floatSample = (float *)malloc(sizeof(float) * sampleLength);
    doubleSample = (double *)malloc(sizeof(double) * sampleLength);

    for (i = 0; i < nRuns; i++)
    {
        double toDeviceTime, kernelTime, fromDeviceTime;

        run_float_test(N, nKernelIters, blocksPerGrid, threadsPerBlock,
                       &toDeviceTime, &kernelTime, &fromDeviceTime,
                       floatSample, sampleLength);
        meanFloatToDeviceTime += toDeviceTime;
        meanFloatKernelTime += kernelTime;
        meanFloatFromDeviceTime += fromDeviceTime;

        run_double_test(N, nKernelIters, blocksPerGrid, threadsPerBlock,
                        &toDeviceTime, &kernelTime, &fromDeviceTime,
                        doubleSample, sampleLength);
        meanDoubleToDeviceTime += toDeviceTime;
        meanDoubleKernelTime += kernelTime;
        meanDoubleFromDeviceTime += fromDeviceTime;

        if (i == 0 || i == nRuns - 1) // check diff between float and double
        {
            int j;
            printf("Iter = %d\n", i);
            printf("Input\tDiff Between Single- and Double-Precision\n");
            printf("------\t------\n");

            for (j = 0; j < sampleLength; j++)
            {
                printf("%d\t%.20e\n", j,
                       fabs(doubleSample[j] - (double)floatSample[j]));
            }

            printf("\n");
        }
    }

    meanFloatToDeviceTime /= nRuns;
    meanFloatKernelTime /= nRuns;
    meanFloatFromDeviceTime /= nRuns;
    meanDoubleToDeviceTime /= nRuns;
    meanDoubleKernelTime /= nRuns;
    meanDoubleFromDeviceTime /= nRuns;

    printf("For single-precision floating point, mean times for:\n");
    printf("  Copy to device:   %f s\n", meanFloatToDeviceTime);
    printf("  Kernel execution: %f s\n", meanFloatKernelTime);
    printf("  Copy from device: %f s\n", meanFloatFromDeviceTime);
    printf("For double-precision floating point, mean times for:\n");
    printf("  Copy to device:   %f s (%.2fx slower than single-precision)\n",
           meanDoubleToDeviceTime,
           meanDoubleToDeviceTime / meanFloatToDeviceTime);
    printf("  Kernel execution: %f s (%.2fx slower than single-precision)\n",
           meanDoubleKernelTime,
           meanDoubleKernelTime / meanFloatKernelTime);
    printf("  Copy from device: %f s (%.2fx slower than single-precision)\n",
           meanDoubleFromDeviceTime,
           meanDoubleFromDeviceTime / meanFloatFromDeviceTime);

    cudaFree(warm_up_i);
    free(h_tmp);
    return 0;
}
