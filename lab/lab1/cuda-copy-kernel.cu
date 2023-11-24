#include <cstdio>
#include <iostream>
#include "cuda.h"

using namespace std;

__global__ void cudaCopyByBlocks(float *tab0, const float *tab1, int size)
{
  int idx = blockIdx.x;
  if (idx < size) { tab0[idx] = tab1[idx]; }
}

__global__ void cudaCopyByBlocksThreads(float *tab0, const float *tab1, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) { tab0[idx] = tab1[idx]; }
}

int main(int argc, char **argv) {
  float *A, *B, *dA, *dB;
  int N, i;

  if (argc < 2) {
    printf("Usage: %s N\n", argv[0]);
    return 0;
  }
  N = atoi(argv[1]);

  // Initialization
  // Initialisation
  A = (float *) malloc(sizeof(float) * N);
  B = (float *) malloc(sizeof(float) * N);
  for (i = 0; i < N; i++) { 
    A[i] = (float)i;
    B[i] = 0.0f;
  }
  
  // Allocate dynamic arrays dA and dB of size N on the GPU with cudaMalloc
  // Allouer les tableau dA et dB dynamiques de size N sur le GPU avec cudaMalloc 
  cudaError_t cuStat;
  cuStat = cudaMalloc((void **)&dA, N*sizeof(float));
  if (cuStat != cudaSuccess) {
    printf("L'allocation de la memoire a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
  }
  cuStat = cudaMalloc((void **)&dB, N*sizeof(float));
  if (cuStat != cudaSuccess) {
    printf("L'allocation de la memoire a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
  }

  // Copy A into dA and B into dB
  // Copier A dans dA et B dans dB
  cudaMemcpy(dA, A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, N*sizeof(float), cudaMemcpyHostToDevice);

  // Copy dA into dB using the kernel cudaCopyByBlocks
  // Copier dA dans dB avec le kernel cudaCopyByBlocks
  cudaCopyByBlocks<<<N, 1>>>(dB, dA, N);

  // Wait for kernel cudaCopyByBlocks to finish
  // Attendre que le kernel cudaCopyByBlocks termine
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("Kernel execution failed with error: \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Copy dB into B for verification
  // Copier dB dans B pour la verification
  cudaMemcpy(B, dB, N*sizeof(float), cudaMemcpyDeviceToHost);

  // Verify the results on the CPU by comparing B with A
  // Verifier le resultat en CPU en comparant B avec A
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "La copie est incorrecte!\n"; }
  else { cout << "La copie est correcte!\n"; }

  // Reinitialize B to zero, then copy B into dB again to test the second copy kernel
  // Remettre B a zero puis recopier dans dB tester le deuxieme kernel de copie
  for (int i = 0; i < N; i++) { B[i] = 0.0f; }
  cudaMemcpy(dB, B, N*sizeof(float), cudaMemcpyHostToDevice);

  // Copy dA into dB with the kernel cudaCopyByBlocksThreads
  // Copier dA dans dB avec le kernel cudaCopyByBlocksThreads
  unsigned nbThreadsPerBlock = 32;
  unsigned nbBlocks = N%nbThreadsPerBlock ? N/nbThreadsPerBlock + 1 : N/nbThreadsPerBlock;
  cudaCopyByBlocksThreads<<<nbBlocks, nbThreadsPerBlock>>>(dB, dA, N);


  // Wait for the kernel cudaCopyByBlocksThreads to finish
  // Attendre que le kernel cudaCopyByBlocksThreads termine
  cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("L'execution du kernel a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Copy dB into B for verification
  // Copier dB dans B pour la verification
  cudaMemcpy(B, dB, N*sizeof(float), cudaMemcpyDeviceToHost);

  // Verify the results on the CPU by comparing B with A
  // Verifier le resultat en CPU en comparant B avec A
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "La copie est incorrecte!\n"; }
  else { cout << "La copie est correcte!\n"; }

  // Deallocate arrays dA[N] and dB[N] on the GPU
  // Desaollouer le tableau dA[N] et dB[N] sur le GPU
  cuStat = cudaFree(dA);
  if (cuStat != cudaSuccess) {
    printf("La libération de la memoire a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
  }
  cuStat = cudaFree(dB);
  if (cuStat != cudaSuccess) {
    printf("La libération de la memoire a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
  }

  // Deallocate A and B
  // Desallouer A et B
  free(A);
  free(B);

  return 0;
}
