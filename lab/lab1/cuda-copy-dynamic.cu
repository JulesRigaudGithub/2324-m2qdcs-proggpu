#include <cstdio>
#include <iostream>
#include "cuda.h"

using namespace std;

int main(int argc, char **argv) {
  float *A, *B, *dA;
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
  for (i = 0; i < N; i++) { A[i] = (float)i; }
  
  // Allocate the dynamic float array dA[N] on the GPU using cudaMalloc
  // Allouer le tableau dA dynamique de taille N sur le GPU avec cudaMalloc 
  cudaError_t cuStat;
  cuStat = cudaMalloc((void **)&dA, N*sizeof(float));
  if (cuStat != cudaSuccess) {
    printf("L'allocation de la memoire a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
  }

  // cudaMemcpy from A[N] to dA[N]
  // cudaMemcpy de A[N] vers dA[N]
  cudaMemcpy(dA, A, N*sizeof(float), cudaMemcpyHostToDevice);

  // cudaMemcpy from dA[N] to B[N]
  // cudaMemcpy de dA[N] vers B[N]
  cudaMemcpy(B, dA, N*sizeof(float), cudaMemcpyDeviceToHost);

  // Desaollouer le tableau dA[N] sur le GPU
  cuStat = cudaFree(dA);
  if (cuStat != cudaSuccess) {
    printf("La libération de la memoire a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cuStat));
  }

  // Attendre que les kernels GPUs terminent
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("L'execution du kernel a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Verify the result
  // Verifier le resultat
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "La copie est incorrecte!\n" << i; }
  else { cout << "La copie est correcte!\n"; }
  free(A);
  free(B);

  return 0;
}
