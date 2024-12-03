#include <cuda_runtime.h>

__global__ void foo(float *ptr)
{
    *ptr = (*ptr) * (*ptr) + (*ptr);
}