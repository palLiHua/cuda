#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define N 1024

int main(int argc, char **argv)
{
    int *restrict A = (int*)malloc(sizeof(int) * N);
    int *restrict B = (int*)malloc(sizeof(int) * N);
    int *restrict C = (int*)malloc(sizeof(int) * N);
    int *restrict D = (int*)malloc(sizeof(int) * N);

    for (int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = 2 * i;
    }

    memset(D, 0, sizeof(int) * N);

#pragma acc kernels if(0 > 1)
    {
        for(int i = 0; i < N; i++)
        {
            C[i] = A[i] + B[i];
        }
        
        for (int i = 0;i < N; i++)
        {
            D[i] = C[i] * A[i];
        }
    }

    for (int i = 0; i < 10; i++)
    {
        printf("%d ", D[i]);
    }

    printf("...\n");
    return 0;
}