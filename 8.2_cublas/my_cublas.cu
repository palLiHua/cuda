#include "../include/util.h"
// #include "util.h"
// #include "cublas.h"
#include "cublas_v2.h"
int rows = 1024;
int cols = 1024;
int check_rows = 1;

float alpha = 3.0f;
float beta = 4.0f;

void check_res(float *A, float *X, float *Y, float *res)
{
    int flag = 0;
    for (int i = 0; i < check_rows; i++)
    {
        float tmp = 0.0;
        for (int j = 0; j < cols; j++)
        {
            tmp += A[j * rows + i] * X[j];
        }
        tmp = alpha * tmp + beta * Y[i];
        if (abs(tmp - res[i]) > 0.01)
        {
            printf("%dth element doesn't match, it should be %f, but get %f\n", i, tmp, res[i]);
            flag = 1;
            break;
        }
    }
    if (!flag) printf("Check %d elements successfully\n", check_rows);
}

void generate_random_vector(int N, float **outX)
{
    int i;
    double rMax = (double)RAND_MAX;
    float *X = (float *)malloc(sizeof(float) * N);

    for (i = 0; i < N; i++)
    {
        int r = rand();
        double dr = (double)r;
        X[i] = (dr / rMax) * 100.0;
    }

    *outX = X;
}

/**
 * Column-major
 */
void generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < M; i++)
        {
            double dr = (double)rand();
            A[j * M + i] = (dr / rMax) * 100.0;
        }
    }

    *outA = A;
}

int main(int argc, char **argv)
{
    srand(9384);

    cublasHandle_t handle = 0;
    
    float *A, *X, *Y;
    float *dA, *dX, *dY;
    float *resY;

    int size_ele = sizeof(float);
    generate_random_dense_matrix(rows, cols, &A);
    generate_random_vector(cols, &X);
    generate_random_vector(rows, &Y);
    generate_random_vector(rows, &resY);

// #ifdef DEBUG
//     for (int i = 0; i < cols; i++)
//     {
//         for (int j = 0; j < rows; j++)
//         {
//             A[i*rows + j] = (float)j;
//             Y[j] = (float)j;
//         }
//         X[i] = (float)i;
//     }
// #endif

    CHECK_CUBLAS(cublasCreate(&handle));

    CHECK(cudaMalloc((float**)&dA, size_ele * cols * rows));
    CHECK(cudaMalloc((float**)&dX, size_ele * cols));
    CHECK(cudaMalloc((float**)&dY, size_ele * rows));

    CHECK_CUBLAS(cublasSetMatrix(rows, cols, size_ele, A, rows, dA, rows));
    CHECK_CUBLAS(cublasSetVector(cols, size_ele, X, 1, dX, 1));
    CHECK_CUBLAS(cublasSetVector(rows, size_ele, Y, 1, dY, 1));

    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, rows, cols, &alpha, dA, rows, dX, 1,
                             &beta, dY, 1));

    CHECK_CUBLAS(cublasGetVector(rows, size_ele, dY, 1, resY, 1));

#ifdef DEBUG
    printf("Matrix A = \n");
    for (int i = 0; i < check_rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%2.2f ", A[j * rows + i]);
        }
        // printf("%f ", X[i]);
        printf("\n");
    }

    printf("Vector X = \n");
    for (int i = 0; i < cols; i++)
    {
        printf("%2.2f\n", X[i]);
    }

    printf("Vector Y = \n");
    for (int i = 0; i < check_rows; i++)
    {
        printf("%2.2f\n", Y[i]);
    }

#endif

    printf("res = \n");
    for (int i = 0; i < check_rows; i++)
    {
        printf("%2.2f\n", resY[i]);
    }
    check_res(A, X, Y, resY);

    free(A);
    free(X);
    free(Y);
    free(resY);

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dX));
    CHECK(cudaFree(dY));
    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}