/**
  * In this exercise, we will implement GPU kernels for computing the average of 9 points on a 2D array.
  * Dans cet exercice, nous implantons un kernel GPU pour un calcul de moyenne de 9 points sur un tableau 2D.
  *





  * For all kernels: Make necessary memory allocations/deallocations and memcpy in the main.
  * Pour tous les kernels: Effectuer les allocations/desallocations et memcpy necessaires dans le main.
  */

#include <iostream>
#include <cstdio>
#include "cuda.h"
#include "omp.h"

#include <chrono>

#define N 1024
#define K 2
#define BSXY 32

// The matrix is stored by rows, that is A(i, j) = A[i + j * N]. The average should be computed on Aavg array.
// La matrice A est stockee par lignes, a savoir A(i, j) = A[i + j * N]
float *A;
float *Aavg;

// Idem pour les miroirs sur GPU
float *dA;
float *dAavg;

// Reference CPU implementation
// Code de reference pour le CPU
void ninePointAverageCPU()
{
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      Aavg[i + j * N] = (A[i - 1 + (j - 1) * N] + A[i - 1 + (j) * N] + A[i - 1 + (j + 1) * N] +
          A[i + (j - 1) * N] + A[i + (j) * N] + A[i + (j + 1) * N] +
          A[i + 1 + (j - 1) * N] + A[i + 1 + (j) * N] + A[i + 1 + (j + 1) * N]);
    }
  }
}

void verifyResults()
{
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      float avg = 0.0f;

      avg = (A[i - 1 + (j - 1) * N] + A[i - 1 + (j) * N] + A[i - 1 + (j + 1) * N] +
          A[i + (j - 1) * N] + A[i + (j) * N] + A[i + (j + 1) * N] +
          A[i + 1 + (j - 1) * N] + A[i + 1 + (j) * N] + A[i + 1 + (j + 1) * N]);
      if (std::abs(Aavg[i * N + j] - avg) > 1e-6) {
        std::cout << "Multiplication is incorrect for the element Aavg[" << i << "][" << j << "]" << std::endl;
        std::cout << "Difference is " << std::abs(Aavg[i * N + j] - avg) << std::endl;
        return;
      }
    }
  }
  std::cout << "Multiplication is correct!" << std::endl;
}

  // * Kernel 1: Use 1D grid of blocks (only blockIdx.x), no additional threads (1 thread per block)
  // * Kernel 1: Utiliser grille 1D de blocs (seulement blockIdx.x), pas de threads (1 thread par bloc)
  // *
__global__ void kernel1(float *dA, float *dAavg, int n) {

  int i = blockIdx.x % (n - 2) + 1;
  int j = blockIdx.x / (n - 2) + 1;

  dAavg[i + j * n] = (dA[i - 1 + (j - 1) * n] + dA[i - 1 + (j) * n] + dA[i - 1 + (j + 1) * n] +
      dA[i + (j - 1) * n] + dA[i + (j) * n] + dA[i + (j + 1) * n] +
      dA[i + 1 + (j - 1) * n] + dA[i + 1 + (j) * n] + dA[i + 1 + (j + 1) * n]);
}

  // * Kernel 2: Use 2D grid of blocks (blockIdx.x/.y), no additional threads (1 thread per block)
  // * Kernel 2: Utiliser grille 2D de blocs (blockIdx.x/.y), pas de threads (1 thread par bloc)
  // *
__global__ void kernel2(float *dA, float *dAavg, int n) {

  int i = blockIdx.x + 1;
  int j = blockIdx.y + 1;

  dAavg[i + j * n] = (dA[i - 1 + (j - 1) * n] + dA[i - 1 + (j) * n] + dA[i - 1 + (j + 1) * n] +
      dA[i + (j - 1) * n] + dA[i + (j) * n] + dA[i + (j + 1) * n] +
      dA[i + 1 + (j - 1) * n] + dA[i + 1 + (j) * n] + dA[i + 1 + (j + 1) * n]);
}

  // * Kernel 3: Use 2D grid of blocks and 2D threads (BSXY x BSXY), each thread computing 1 element of Aavg
  // * Kernel 3: Utiliser grille 2D de blocs, threads de 2D (BSXY x BSXY), chaque thread calcule 1 element de Aavg
  // *
__global__ void kernel3(float *dA, float *dAavg, int n) {

  int i = threadIdx.x + blockIdx.x*BSXY + 1;
  int j = threadIdx.y + blockIdx.y*BSXY + 1;

  if ((i < n - 1) && (j < n - 1)) { 
  dAavg[i + j * n] = (dA[i - 1 + (j - 1) * n] + dA[i - 1 + (j) * n] + dA[i - 1 + (j + 1) * n] +
      dA[i + (j - 1) * n] + dA[i + (j) * n] + dA[i + (j + 1) * n] +
      dA[i + 1 + (j - 1) * n] + dA[i + 1 + (j) * n] + dA[i + 1 + (j + 1) * n]);
  }
}

  // * Kernel 4: Use 2D grid of blocks and 2D threads, each thread computing 1 element of Aavg, use shared memory. Each block should load BSXY x BSXY elements of A, then compute (BSXY - 2) x (BSXY - 2) elements of Aavg. Borders of tiles loaded by different blocks must overlap to be able to compute all elements of Aavg.
  // * Kernel 4: Utiliser grille 2D de blocs, threads de 2D, chaque thread calcule 1 element de Aavg, avec shared memory. Chaque bloc doit lire BSXY x BSXY elements de A, puis calculer avec ceci (BSXY - 2) x (BSXY - 2) elements de Aavg. Les bords des tuiles chargees par de differents blocs doivent chevaucher afin de pouvoir calculer tous les elements de Aavg.
  // *
__global__ void kernel4(float *dA, float *dAavg, int n) {

  __shared__ float shA[BSXY][BSXY];

  int i = threadIdx.x + blockIdx.x*(BSXY-2);
  int j = threadIdx.y + blockIdx.y*(BSXY-2);

  shA[threadIdx.y][threadIdx.x] = dA[i + j*n];

  if ((threadIdx.x != 0) && (threadIdx.x != BSXY - 1)
      && (threadIdx.y != 0) && (threadIdx.y != BSXY - 1)
      && (i < n - 1) && (j < n - 1)) {
              dAavg[i + j * n] = (shA[threadIdx.y - 1][threadIdx.x - 1] + shA[threadIdx.y][threadIdx.x - 1] + shA[threadIdx.y + 1][threadIdx.x - 1] +
                  shA[threadIdx.y - 1][threadIdx.x] + shA[threadIdx.y][threadIdx.x] + shA[threadIdx.y + 1][threadIdx.x] +
                  shA[threadIdx.y - 1][threadIdx.x + 1] + shA[threadIdx.y][threadIdx.x + 1] + shA[threadIdx.y + 1][threadIdx.x + 1]);
  }
}

  // * Kernel 5: Use 2D grid of blocks and 2D threads, use shared memory, each thread computes KxK elements of Aavg
  // * Kernel 5: Utiliser grille 2D de blocs, threads de 2D, avec shared memory et KxK ops par thread
  // *
__global__ void kernel5(float *dA, float *dAavg, int n) {

  __shared__ float shA[K*BSXY][K*BSXY];

  int i = threadIdx.x + blockIdx.x*K*BSXY;
  int j = threadIdx.y + blockIdx.y*K*BSXY;

  for (int i_bis = 0 ; i < K ; i++) {
    for (int j_bis = 0 ; j < K ; j++) {
      shA[threadIdx.y + BSXY*j_bis][threadIdx.x + BSXY*i_bis] = dA[i + BSXY*i_bis + (j+BSXY*j_bis)*n];
    }
  }
  __syncthreads();

  for (int i_bis = 0 ; i < K ; i++) {
    for (int j_bis = 0 ; j < K ; j++) {
      if ((threadIdx.x + BSXY*i_bis != 0) && (threadIdx.x + BSXY*i_bis != (K*BSXY - 1))
          && (threadIdx.y + BSXY*j_bis != 0) && (threadIdx.y + BSXY*j_bis != (K*BSXY - 1))
          && (i + BSXY*i_bis < n - 1) && (j + BSXY*j_bis < n - 1)) {
              dAavg[i + BSXY*i_bis + (j + BSXY*j_bis) * n] = (shA[threadIdx.y + BSXY*j_bis - 1][threadIdx.x + BSXY*i_bis - 1] + shA[threadIdx.y + BSXY*j_bis][threadIdx.x + BSXY*i_bis - 1] + shA[threadIdx.y + BSXY*j_bis + 1][threadIdx.x + BSXY*i_bis - 1] +
                  shA[threadIdx.y + BSXY*j_bis - 1][threadIdx.x + BSXY*i_bis] + shA[threadIdx.y + BSXY*j_bis][threadIdx.x + BSXY*i_bis] + shA[threadIdx.y + BSXY*j_bis + 1][threadIdx.x + BSXY*i_bis] +
                  shA[threadIdx.y + BSXY*j_bis - 1][threadIdx.x + BSXY*i_bis + 1] + shA[threadIdx.y + BSXY*j_bis][threadIdx.x + BSXY*i_bis + 1] + shA[threadIdx.y + BSXY*j_bis + 1][threadIdx.x + BSXY*i_bis + 1]);
      }
    }
  }
}


int main()
{
  // Initialisation
  A = (float *) malloc (N * N * sizeof(float));
  Aavg = (float *) malloc (N * N * sizeof(float));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i + j * N] = (float)i * (float)j;
    }
  }

  // Allocate dA and dAavg, then copy the array A to the GPU
  // Allouer dA et dAavg, puis copier le tableau A vers le GPU
  cudaMalloc(&dA, N * N * sizeof(float));
  cudaMalloc(&dAavg, N * N * sizeof(float));

  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

  /* Lancement du kernel */

  auto start = std::chrono::high_resolution_clock::now();

  // Call each GPU kernel appropriately to multiply matrices A and B
  // Measure and print the execution time and performance (GFlops/s) of each kernel, without counting the data transfer time
  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.x = (N-2)*(N-2);
    dimGrid.y = 1;
    dimGrid.z = 1;
    // kernel1<<<dimGrid, 1>>>(dA, dAavg, N);
  }
  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimBlock.x = 1;
    dimBlock.y = 1;
    dimBlock.z = 1;
    dimGrid.x = N-2;
    dimGrid.y = N-2;
    dimGrid.z = 1;
    // kernel2<<<dimGrid, 1>>>(dA, dAavg, N);
  }
  { 
    dim3 dimGrid;
    dim3 dimBlock;
    dimBlock.x = BSXY;
    dimBlock.y = BSXY;
    dimBlock.z = 1;
    dimGrid.x = (N - 2 - 1)/BSXY + 1;
    dimGrid.y = (N - 2 - 1)/BSXY + 1;
    dimGrid.z = 1;
    // kernel3<<<dimGrid, dimBlock>>>(dA, dAavg, N);
  }
  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimBlock.x = BSXY;
    dimBlock.y = BSXY;
    dimBlock.z = 1;
    dimGrid.x = (N - 2 - 1)/(BSXY-2) + 1;
    dimGrid.y = (N - 2 - 1)/(BSXY-2) + 1;
    dimGrid.z = 1;
    kernel4<<<dimGrid, dimBlock>>>(dA, dAavg, N);
  }
  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimBlock.x = BSXY;
    dimBlock.y = BSXY;
    dimBlock.z = 1;
    dimGrid.x = (N - 2 - 1)/(K*BSXY-2) + 1;
    dimGrid.y = (N - 2 - 1)/(K*BSXY-2) + 1;
    dimGrid.z = 1;
    // kernel5<<<dimGrid, dimBlock>>>(dA, dAavg, N); // Je n'ai pas réussi à faire fonctionner le kernel 5
  }


  // Copy the array dC back to the CPU
  // Recopier le tableau dC vers le CPU
  cudaMemcpy(Aavg, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  auto stop = std::chrono::high_resolution_clock::now();

  // Estimation du temps de transfert retour pour déduction
  auto tr_start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(Aavg, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  auto tr_stop = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration<double>((stop - start) - (tr_stop - tr_start)).count();
  std::cout << "Le temps d'exécution du kernel est de " << duration << " secondes\n";

  double gflops_s = (2.0*pow(N,3))/duration*1E-9;
  std::cout << "Cela correspond à " << gflops_s << "GFlops/s" << std::endl;

  // Verify the results
  // Verifier les resultats
  // ninePointAverageCPU();
  verifyResults();


  free(A);
  free(Aavg);

  // Deallocate dA, dAavg
  // Desallouer dA, dAavg
  cudaFree(dA); cudaFree(dAavg);

  return 0;
}
