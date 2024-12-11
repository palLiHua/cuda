#include <stdio.h>
#include <stdlib.h>

#define N 1024

int main(int argc, char **argv)
{
    int i = 0;
    int *A = (int*)malloc(sizeof(int) * N);
    int *B = (int*)malloc(sizeof(int) * N);
    int *C = (int*)malloc(sizeof(int) * N);
    int *D = (int*)malloc(sizeof(int) * N);

    for (i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = 2 * i;
    }

#pragma acc parallel copyin(A[0:N], B[0:N]) copyout(C[0:N], D[0:N])
    {
#pragma acc loop
        for (i = 0; i < N; i++)
        {
            C[i] = A[i] + B[i];
        }

#pragma acc loop
        for (i = 0; i < N; i++)
        {
            D[i] = C[i] * A[i];
        }
    }

/*
 *  Same as below 
 */

// #pragma acc data copyin(A[0:N], B[0:N]) copyout(C[0:N], D[0:N])
//     {
// #pragma acc parallel
//         {
// #pragma acc loop
//             for (i = 0; i < N; i++)
//             {
//                 C[i] = A[i] + B[i];
//             }

// #pragma acc loop
//             for (i = 0; i < N; i++)
//             {
//                 D[i] = C[i] * A[i];
//             }
//         }
//     }

    for (i = 0; i < 10; i++)
    {
        printf("%d ", D[i]);
    }
    printf("...\n");

    return 0;
}