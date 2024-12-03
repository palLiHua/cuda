#include "../include/util.h"

__device__ float myAtomicAdd(float *address, float incr) 
{
    unsigned int *typedAddress = (unsigned int *)address;
    float currentVal = *address;
    printf("currentVal = %f\n", currentVal);
    printf("incr = %f\n", incr);
    unsigned int expected = __float2uint_rn(currentVal);
    unsigned int desired = __float2uint_rn(currentVal + incr);
    unsigned int oldIntValue = atomicCAS(typedAddress, expected, desired); 
    printf("old = %d\n", oldIntValue);
    while (oldIntValue != expected)
    {
        expected = oldIntValue;
        desired = __float2uint_rn(__uint2float_rn(oldIntValue) + incr);
        oldIntValue = atomicCAS(typedAddress, expected, desired); 
    }
    return __uint2float_rn(oldIntValue); 
}

__global__ void kernel(float* add, float incr)
{
    int tid = threadIdx.x;
    if (tid == 0)
    {
        myAtomicAdd(add, incr);
    }
}

int main(int argc, char** argv)
{
    float* add = NULL;
    CHECK(cudaMalloc((float**)&add,sizeof(float)));
    
    float incr = 17.15;
    float res = 12.3;
    CHECK(cudaMemcpy(add, &res, sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(32);
    dim3 grid(1);
    kernel<<<grid, block>>>(add, incr);
    CHECK(cudaMemcpy(&res, add, sizeof(float), cudaMemcpyDeviceToHost));
    printf("res = %f\n", res);
    return 0;
}
