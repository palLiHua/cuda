#ifndef UTIL_H
#define UTIL_H
#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
    }\
}\

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#ifdef _WIN32
int gettimeofday(struct timeval *tp, void *tzp)
{
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;
  GetLocalTime(&wtm);
  tm.tm_year   = wtm.wYear - 1900;
  tm.tm_mon   = wtm.wMonth - 1;
  tm.tm_mday   = wtm.wDay;
  tm.tm_hour   = wtm.wHour;
  tm.tm_min   = wtm.wMinute;
  tm.tm_sec   = wtm.wSecond;
  tm.tm_isdst  = -1;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 1000;
  return 0;
}
#endif
double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}

void initialDataInt(int *ip, int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (int)(rand()&0xff);
        // ip[i] = i;
    }  
}

void initialData(float *ip, int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
        // ip[i] = i;
    }
}

void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using device %d: %s\n",dev,deviceProp.name);
  CHECK(cudaSetDevice(dev));

}

void checkRes(float *hostRef, float *gpuRef, int size)
{
    int flag = 0;
    double epsilon = 1.0E-4;
    for (int i = 0; i < size; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("Results don\'t match!\n");
            printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n",hostRef[i],i,gpuRef[i],i);
            flag = 1;
            break;
        }
    }
    if (!flag)
    {
        printf("Check result success! All values are identical !\n");
    }
}

#endif