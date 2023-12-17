#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include "cuda.h"
#include <cfloat>

#define BLOCKSIZE 1024

/**
  * Step 1: Write a 1D (blocks and threads) GPU kernel that finds the minimum element in an array dA[N] in each block, then writes the minimum in dAmin[blockIdx.x]. CPU should take this array and find the global minimum by iterating over this array.
  * Etape 1: Ecrire un kernel GPU 1D (blocs et threads) qui trouve l'element minimum d'un tableau dA[N] pour chaque bloc et ecrit le minimum de chaque bloc dans dAmin[blockIdx.x]. En suite, CPU reprend dAmin et calcul le minimum global en sequentiel sur ce petit tableau.
  *
  * Step 2: The first call to findMinimum reduces the size of the array to N/BLOCKSIZE. In this version, use findMinimum a second time on this resulting array, in order to reduce the size to N/(BLOCKSIZE*BLOCKSIZE) so that computation on the CPU to find the global minimum becomes negligible.
  * Etape 2: Le premier appel au findMinimum reduit la taille du tableau a parcourir en sequentiel a N/BLOCKSIZE. Dans cette version, utiliser findMinimum une deuxieme fois afin de reduire la taille du tableau a  N/(BLOCKSIZE*BLOCKSIZE) pour que le calcul cote CPU pour trouver le minimum global devienne negligeable.
  *
  * To find the minimum of two floats on a GPU, use the function fminf(x, y).
  * Pour trouver le minimum des deux flottants en GPU, utiliser la fonction fminf(x, y).
  */

__global__ void findMinimum(float *dA, float *dAmin, int N)
{
  __shared__ volatile float buff[BLOCKSIZE];
  int idx = threadIdx.x + blockIdx.x * BLOCKSIZE;
  buff[threadIdx.x] = dA[idx];

  for (int step = BLOCKSIZE/2; step >= 1 ; step = step / 2) {
    if (threadIdx.x < BLOCKSIZE/2) {
      float a = buff[threadIdx.x];
      float b = buff[2*threadIdx.x];
      buff[threadIdx.x] = fminf(a, b);
    }
  }

  if (threadIdx.x == 0) { dAmin[blockIdx.x] = buff[0]; }
}

using namespace std;

int main()
{
  srand(1234);
  int N = 100000000;
  int numBlocks = N/ BLOCKSIZE;// = ???; (TODO / A FAIRE ...)
  float *A, *dA; // Le tableau dont minimum on va chercher
  float *Amin, *dAmin; // Amin contiendra en suite le tableau reduit par un facteur de BLOCKSIZE apres l'execution du kernel GPU

  // Allocate arrays A[N] and Amin[numBlocks} on the CPU
  A = (float *)malloc(sizeof(float) * N);
  Amin = (float *)malloc(sizeof(float) * numBlocks);

  // Allocate arrays dA[N] and dAmin[numBlocks} on the GPU
  cudaMalloc(&dA, sizeof(float) * N);
  cudaMalloc(&dAmin, sizeof(float) * numBlocks);



  // Initialize the array A, set the minimum to -1
  // Initialiser le tableau A, mettre le minimum a -1.
  for (int i = 0; i < N; i++) { A[i] = (float)(rand() % 1000); }
  int secret = rand() % N;
  cout << secret << endl;
  A[1] = -1.0; 

  // Transfer A on the GPU (dA) with cudaMemcpy
  cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice);

  // Put maximum attainable value to minA.
  // Affecter la valeur maximum atteignable dans minA
  float minA = FLT_MAX; 

  // Find the minimum of the array dA for each thread block, put it in dAMin[...] and transfer to the CPU, then find the global minimum of this smaller array and put it in minA.
  findMinimum<<<numBlocks, BLOCKSIZE>>>(dA, dAmin, N);

  cudaMemcpy(Amin, dAmin, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

  for (int k = 0; k < numBlocks; k++) {
    int test = Amin[k];
    // cout << test << " ";
    minA = (minA < test) ? minA : test;
  }

  // Verify the result
  if (minA == -1) { cout << "The minimum is correct!" << endl; }
  else { cout << "The minimum found (" << minA << ") is incorrect (it should have been -1)!" << endl; }

  return 0;
}
