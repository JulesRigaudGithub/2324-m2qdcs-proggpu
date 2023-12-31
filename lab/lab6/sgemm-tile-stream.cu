/**
  * Compilation: nvcc sgemm-tile-stream.cu -o sgemm-tile-stream -lcublas
  * Execution: ./sgemm-tile-stream
  * 
  * Task 1: Basic CuBLAS execution and benchmarking.
  *   -Allocate and initialize three N * N column-major float matrices A, B, C on the CPU.
  *   -Allocate dA, dB, dC on the GPU.
  *   -Copy contents of A, B to dA, dB
  *   -Execute cublasSgemm(...)
  *   -Copy dC back to C
  *   -Measure and print the total execution time including host-to-device copy, sgemm, and device-to-host copy and flops/s (sgemm performs 2*N*N*(N-1) flops)
  *
  *
  * Task 2: Implementing tiled cublasSgemm with pipelining
  *   -Create one transfer stream for host-to-device transfers and P x P streams for computing each tile of C(pi, pj) for 0 <= pi, pj < P
  *   -Transfer all tiles A(pi, pj) and B(pi, pj) to dA(pi, pj) and dB(pi, pj) in the transfer stream for 0 <= pi, pj < P, and launch an event ea(pi, pj) and eb(pi, pj) for each tile transfer
  *   -Schedule all tile sgemms required to compute dC(pi, pj) into stream(pi, pj), add data dependencies for each operation with event wait. Use cublasSetStream(handle, stream) each time to make sure that sgemm is placed onto the stream(pi, pj).
  *   -Once all sgemms for a tile dC(pi, pj) are completed, copy dC(pi, pj) into the tile C(pi, pj) in the stream (pi, pj).
  *   -Measure and print the total execution time including tile data transfers and sgemm calls, and print flops/s (sgemm performs 2*N*N*(N-1) flops)
  *   -Tune the value of P by experimentation for N=4096.
  */

/** cublasSgemm signature:
  *
  * cublasStatus_t cublasSgemm(
  * cublasHandle_t handle,
  * cublasOperation_t transa,
  * cublasOperation_t transb,
  * int m, int n, int k,
  * const float *alpha,
  * const float *A, int lda,
  * const float *B, int ldb,
  * const float *beta,
  * float *C, int ldc)
  *
  * See https://docs.nvidia.com/cuda/cublas/index.html for details of usage.
  */

/** cudaMemcpy2DAsync signature:
  *
  * cudaError_t cudaMemcpy2DAsync(
  * void* dst,
  * size_t dpitch,
  * const void* src,
  * size_t spitch,
  * size_t width,
  * size_t height,
  * cudaMemcpyKind kind,
  * cudaStream_t stream = 0)
  *
  * See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge529b926e8fb574c2666a9a1d58b0dc1 for details of usage.
  */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>

#define N 4096
#define P 2

static void simple_sgemm(int n, float alpha, const float *A, const float *B,
    float beta, float *C)
{
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float prod = 0;
      for (int k = 0; k < n; ++k) { prod += A[k * n + i] * B[j * n + k]; }
      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}

int main(int argc, char **argv) {
  cublasStatus_t status;
  float *A;
  float *B;
  float *C;
  float *C_ref;
  float *d_A = 0;
  float *d_B = 0;
  float *d_C = 0;
  float alpha = 1.0f;
  float beta = 1.0f;
  int n2 = N * N;
  cublasHandle_t handle;
  cudaError_t cuStat;

  cudaStream_t computeStreams[P][P];
  cudaStream_t transferStream;

  cudaEvent_t eventTileA[P][P];
  cudaEvent_t eventTileB[P][P];

  int tile_size = N / P;

  //  -Create one transfer stream for host-to-device transfers and P x P streams for computing each tile of C(pi, pj) for 0 <= pi, pj < P
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      cudaStreamCreate(&computeStreams[i][j]);
    }
  }
  cudaStreamCreate(&transferStream);

  // Events registration
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      cudaEventCreate(&eventTileA[i][j]);
      cudaEventCreate(&eventTileB[i][j]);
    }
  }

  // CUBLAS init
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS initialization error!\n");
    return 1;
  }

  // TODO / A FAIRE ...
  // Task 1: Basic CuBLAS execution and benchmarking.
  // *   -Allocate and initialize three N * N column-major float matrices A, B, C on the CPU.
  A = (float *) malloc(sizeof(float) * n2);
  B = (float *) malloc(sizeof(float) * n2);
  C = (float *) malloc(sizeof(float) * n2);
 
  // *   -Allocate dA, dB, dC on the GPU.
  cuStat = cudaMalloc((void **)&d_A, sizeof(float) * n2);
  if (cuStat != cudaSuccess) {
    printf("L'allocation de la memoire a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
    exit(1);
  }
  cuStat = cudaMalloc((void **)&d_B, sizeof(float) * n2);
  if (cuStat != cudaSuccess) {
    printf("L'allocation de la memoire a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
    exit(1);
  }
  cuStat = cudaMalloc((void **)&d_C, sizeof(float) * n2);
  if (cuStat != cudaSuccess) {
    printf("L'allocation de la memoire a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
    exit(1);
  }
  cuStat = cudaMemset((void *)d_C, 0, sizeof(float) * n2);
  if (cuStat != cudaSuccess) {
    printf("L'assignation de la memoire a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
    exit(1);
  }

  auto start = std::chrono::high_resolution_clock::now();

  // // *   -Copy contents of A, B to dA, dB
  // cuStat = cudaMemcpy(d_A, A, sizeof(float) * n2, cudaMemcpyHostToDevice);
  // if (cuStat != cudaSuccess) {
  //   printf("Le transfert a échoué avec le message d'erreur %s", cudaGetErrorString(cuStat));
  //   exit(1);
  // }
  // cuStat = cudaMemcpy(d_B, B, sizeof(float) * n2, cudaMemcpyHostToDevice);
  // if (cuStat != cudaSuccess) {
  //   printf("Le transfert a échoué avec le message d'erreur %s", cudaGetErrorString(cuStat));
  //   exit(1);
  // }

  // -Transfer all tiles A(pi, pj) and B(pi, pj) to dA(pi, pj) and dB(pi, pj) in the transfer stream for 0 <= pi, pj < P, and launch an event ea(pi, pj) and eb(pi, pj) for each tile transfer

  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {

      cuStat = cudaMemcpy2DAsync(
        d_A + (i + j*N) * tile_size,
        sizeof(float)*tile_size,
        A + (i + j*N) * tile_size,
        sizeof(float)*tile_size,
        sizeof(float)*tile_size,
        sizeof(float)*tile_size,
        cudaMemcpyHostToDevice,
        transferStream);
      if (cuStat != cudaSuccess) {
        printf("Le transfert a échoué avec le message d'erreur %s", cudaGetErrorString(cuStat));
        exit(1);
      }
      cudaEventRecord(eventTileA[i][j], transferStream);
      
      cuStat = cudaMemcpy2DAsync(
        d_B + (i + j*N) * tile_size,
        sizeof(float)*tile_size,
        B + (i + j*N) * tile_size,
        sizeof(float)*tile_size,
        sizeof(float)*tile_size,
        sizeof(float)*tile_size,
        cudaMemcpyHostToDevice,
        transferStream);
      if (cuStat != cudaSuccess) {
        printf("Le transfert a échoué avec le message d'erreur %s", cudaGetErrorString(cuStat));
        exit(1);
      }
      cudaEventRecord(eventTileB[i][j], transferStream);
    }
  }


  // -Schedule all tile sgemms required to compute dC(pi, pj) into stream(pi, pj), add data dependencies for each operation with event wait. Use cublasSetStream(handle, stream) each time to make sure that sgemm is placed onto the stream(pi, pj).
  // *   -Execute cublasSgemm(...)
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      cublasSetStream(handle, computeStreams[i][j]);
      for (int k = 0; k < P; k++) {
        // add A(i, k) * B(k, j) to C(i,j)
        cuStat = cudaStreamWaitEvent(computeStreams[i][j], eventTileA[i][k]);
        if (cuStat != cudaSuccess) {
          printf("L'attente de l'événement à échoué %s", cudaGetErrorString(cuStat));
          exit(1);
        }
        cuStat = cudaStreamWaitEvent(computeStreams[i][j], eventTileB[k][j]);
        if (cuStat != cudaSuccess) {
          printf("L'attente de l'événement à échoué %s", cudaGetErrorString(cuStat));
          exit(1);
        }
        status = cublasSgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          tile_size, tile_size, tile_size,
          &alpha,
          d_A + (i + k*N) * tile_size, tile_size,
          d_B + (k + j*N) * tile_size, tile_size,
          &beta,
          d_C + (i + j*N) * tile_size, N);
        if (status != CUBLAS_STATUS_SUCCESS) {
          fprintf(stderr, "CUBLAS Sgemm error!\n");
          return 1;
        }
      }

      cuStat = cudaMemcpy2DAsync(
        C + (i + j*N) * tile_size,
        sizeof(float)*tile_size,
        d_C + (i + j*N) * tile_size,
        sizeof(float)*tile_size,
        sizeof(float)*tile_size,
        sizeof(float)*tile_size,
        cudaMemcpyDeviceToHost,
        computeStreams[i][j]);
      if (cuStat != cudaSuccess) {
        printf("Le transfert a échoué avec le message d'erreur %s", cudaGetErrorString(cuStat));
        exit(1);
      }
      cudaEventRecord(eventTileA[i][j], transferStream);
    }
  }


  // *   -Copy dC back to C
  cuStat = cudaMemcpy(C, d_C, sizeof(float) * n2, cudaMemcpyDeviceToHost);
  if (cuStat != cudaSuccess) {
    printf("Le transfert a échoué avec le message d'erreur %s", cudaGetErrorString(cuStat));
    exit(1);
  }

  auto stop = std::chrono::high_resolution_clock::now();
  
  // *   -Measure and print the total execution time including host-to-device copy, sgemm, and device-to-host copy and flops/s (sgemm performs 2*N*N*(N-1) flops)
  auto duration = std::chrono::duration<double>(stop - start).count();
  std::cout << "Le temps d'exécution du kernel est de " << duration << " secondes\n";

  double gflops_s = (2.0*pow(N,2)*(N-1))/duration*1E-9;
  std::cout << "Cela correspond à " << gflops_s << "GFlops/s" << std::endl;

  // // Synchronization with the device
  // cuStat = cudaDeviceSynchronize();
  // if (cuStat != cudaSuccess) {
  //   printf("L'execution du kernel a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
  //   exit(1);
  // }

  // Free GPU memory
  cuStat = cudaFree(d_A);
  if (cuStat != cudaSuccess) {
    printf("La libération de la memoire a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
    exit(1);
  }
  cuStat = cudaFree(d_B);
  if (cuStat != cudaSuccess) {
    printf("La libération de la memoire a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
    exit(1);
  }
  cuStat = cudaFree(d_C);
  if (cuStat != cudaSuccess) {
    printf("La libération de la memoire a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
    exit(1);
  }

  // Free CPU memory
  free(A); free(B); free(C);

  // CUBLAS destroy
  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS shutdown error!\n");
    return 1;
  }

  return 0;
}
