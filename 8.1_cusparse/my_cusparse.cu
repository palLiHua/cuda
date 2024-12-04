#include "../include/util.h"
#include <cusparse.h> 

int rows = 4;
int cols = 4;
int check_rows = 4;

void check_res(float *A, float *X, float *Y)
{
    int flag = 0;
    for (int i = 0; i < check_rows; i++)
    {
        float tmp = 0.0;
        for (int j = 0; j < cols; j++)
        {
            tmp += A[i*cols + j] * X[j];
        }
        if (abs(tmp - Y[i]) > 0.0001)
        {
            printf("%dth element doesn't match, it should be %f, but get %f\n", i, tmp, Y[i]);
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

int generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);
    int totalNnz = 0;

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < M; i++)
        {
            int r = rand();
            float *curr = A + (j * M + i);

            if (r % 3 > 0)
            {
                *curr = 0.0f;
            }
            else
            {
                double dr = (double)r;
                *curr = (dr / rMax) * 100.0;
            }

            if (*curr != 0.0f)
            {
                totalNnz++;
            }
        }
    }

    *outA = A;
    return totalNnz;
}

int main(int argc, char **argv)
{
    srand(9384);

    cusparseHandle_t handle = 0;

    float* A;
    float* X;
    float* Y;

    float *dA, *dX, *dY;
    
    int trueNnz = generate_random_dense_matrix(rows, cols, &A); // M * N
    generate_random_vector(cols, &X);   // N * 1
    generate_random_vector(rows, &Y);   // M * 1

    CHECK_CUSPARSE(cusparseCreate(&handle));

    CHECK(cudaMalloc((float**)&dA, sizeof(float) * rows * cols));
    CHECK(cudaMalloc((float**)&dX, sizeof(float) * cols));
    CHECK(cudaMalloc((float**)&dY, sizeof(float) * rows));

    CHECK(cudaMemcpy(dA, A, sizeof(float) * rows * cols, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dX, X, sizeof(float) * cols, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dY, Y, sizeof(float) * rows, cudaMemcpyHostToDevice));

    /**
     * convert data format
     */
    // cusparseMatDescr_t descrA;
    cusparseDnMatDescr_t descrA;
    // int *dNnzPerRow;
    // int totalNnz;
    // CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseCreateDnMat(&descrA, rows, cols, cols, dA, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, rows, cols, descrA, dA,
    //                             rows, dNnzPerRow, &totalNnz)); // count the number of non-zero

    // if (totalNnz != trueNnz)
    // {
    //     fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
    //             "value: expected %d but got %d\n", trueNnz, totalNnz);
    //     return 1;
    // }

    cusparseSpMatDescr_t descrB;
    int *d_csr_offset;
    size_t bufferSize = 0;
    CHECK(cudaMalloc((int**)&d_csr_offset, sizeof(int) * (rows + 1)));
    CHECK_CUSPARSE(cusparseCreateCsr(&descrB, rows, cols, 0, d_csr_offset, 
                                    NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                    handle, descrA, descrB,
                                    CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                    &bufferSize) );
    void *dBuffer = NULL;
    CHECK( cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, descrA, descrB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) );
    // int64_t num_rows_tmp, num_cols_tmp, nnz;
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(descrB, &num_rows_tmp, &num_cols_tmp,
                                         &nnz) );
    if (nnz != trueNnz)
    {
        printf("Difference detected between cuSPARSE NNZ and true "
                "value: expected %d but got %ld\n", trueNnz, nnz);
        return 1;
    }
    float *d_csr_values;
    int *d_csr_columns;
    CHECK( cudaMalloc((void**) &d_csr_columns, nnz * sizeof(int))   );
    CHECK( cudaMalloc((void**) &d_csr_values,  nnz * sizeof(float)) );

    CHECK_CUSPARSE( cusparseCsrSetPointers(descrB, d_csr_offset, d_csr_columns,
                                           d_csr_values) );
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, descrA, descrB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) );

    /**
     * check convert result
     */
    int *offset = (int*)malloc(sizeof(int) * (rows + 1));
    int *idx = (int*)malloc(sizeof(int) * nnz);
    float *values = (float*)malloc(sizeof(float) * nnz);
    
    CHECK(cudaMemcpy(offset, d_csr_offset, sizeof(int) * (rows + 1), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(idx, d_csr_columns, sizeof(int) * nnz, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(values, d_csr_values, sizeof(float) * nnz, cudaMemcpyDeviceToHost));

    for (int i = 0; i < nnz; i++)
    {
        printf("i = %d, idx = %d, values = %f\n", i, idx[i], values[i]);
    }
    for (int i = 0; i < rows + 1; i++)
    {
        printf("rows = %d, offset = %d\n", i, offset[i]);
    }
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", A[i*cols + j]);
        }
        printf("%f ", X[i]);
        printf("\n");
    }
    free(idx);
    free(offset);
    free(values);
    /**
     * A * X
     */
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseDnVecDescr_t vecX, vecY;
    size_t bufferSize_ = 0;
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, cols, dX, CUDA_R_32F) );
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, rows, dY, CUDA_R_32F) );
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, descrB, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize_) );
    void *dBuffer_ = NULL;
    CHECK( cudaMalloc(&dBuffer_, bufferSize_) );
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, descrB, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) );

    /**
     * copy back result
     */

    CHECK(cudaMemcpy(Y, dY, sizeof(float) * rows, cudaMemcpyDeviceToHost));
    
    // for (int i = 0; i < rows; i++)
    // {
    //     printf("%2.2f\n", Y[i]);
    // }
    check_res(A, X, Y);


    CHECK(cudaFree(dA));
    CHECK(cudaFree(dX));
    CHECK(cudaFree(dY));
    CHECK(cudaFree(dBuffer));
    CHECK(cudaFree(dBuffer_));
    CHECK(cudaFree(d_csr_offset));
    CHECK(cudaFree(d_csr_values));
    CHECK(cudaFree(d_csr_columns));

    // CHECK(cudaFree(dNnzPerRow));

    free(A);
    free(X);
    free(Y);

    CHECK_CUSPARSE( cusparseDestroyDnMat(descrA) );
    CHECK_CUSPARSE( cusparseDestroySpMat(descrB) );
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) );
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) );
    CHECK_CUSPARSE( cusparseDestroy(handle) );

    return 0;
}