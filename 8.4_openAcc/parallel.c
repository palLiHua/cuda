#include <stdlib.h>
#include <stdio.h>

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

#pragma acc parallel
    {
#pragma acc loop
        for (int i = 0; i < N; i++)
        {
            C[i] = A[i] + B[i];
        }

#pragma acc loop
        for (int i = 0; i < N; i++)
        {
            D[i] = C[i] * A[i];
        }
    }

    for (int i = 0; i < 10; i++)
    {
        printf("%d ", D[i]);
    }

    printf("\n");
    return 0;
}