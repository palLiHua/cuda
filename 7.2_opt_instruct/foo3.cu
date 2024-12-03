__global__ void foo(float *ptr) 
{
    *ptr = __fmul_rn(*ptr, *ptr) + *ptr;
}

__global__ void foo2(float *ptr)
{
    *ptr = (*ptr) * (*ptr) + *ptr;
}