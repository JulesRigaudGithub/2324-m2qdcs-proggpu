#include <cstdio>
#include <cuda.h>
#include <iostream>

int N = 1024;
const int nStreams = 4;
float *A, *B, *C;
float *dA, *dB, *dC;
cudaStream_t streams[nStreams];

// Kernel that performs the matrix vector multiplication b(i) = sum_j(A(i, j), x(j))
// A is row-major (stored row-by-row in memory)
__global__ void matvec(float *A, float *x, float *b, int n)
{
  // TODO / A FAIRE ...
  int i = threadIdx.x;
  float c = 0;

  for (int j = 0; j < n ; j++) {
    c += A[i*n + j]*x[j];
  }

  b[i] = c;
}

void verifyResults()
{
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      float c = 0.0f;
      for (int k = 0; k < N; k++) {
        c += A[i * N + k] * B[k + j * N];
      }
      if (std::abs(C[i + j*N] - c) > 1e-6) {
        std::cout << "Multiplication is incorrect for the element C[" << i << "][" << j << "]" << std::endl;
        return;
      }
    }
  }
  std::cout << "Multiplication is correct!" << std::endl;
}

int main()
{
  // A is stored by rows, A(i, j) = A[i * N + j]
  A = (float *) malloc (N * N * sizeof(float));
  // B and C are stored by columns, B(i, j) = B[i + j * N]
  B = (float *) malloc (N * N * sizeof(float));
  C = (float *) malloc (N * N * sizeof(float));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = i + j; // A(i, j) = i + j
      B[i + j * N] = i - j; // B(i, j) = i - j
      C[i + j * N] = 0; // C(i, j) = 0
    }
  }
  cudaMalloc(&dA, N * N * sizeof(float));
  cudaMalloc(&dB, N * nStreams * sizeof(float));
  cudaMalloc(&dC, N * nStreams * sizeof(float));

  // Only copy the entire matrix A. For B and C, they need to be copied and computed one column vector at a time in a streaming manner
  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

  // Create streams
  for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // Compute the matrix-vector multiplication C(:, j) = A * B(:, j) column-by-column using nStreams streams
  for (int j = 0; j < N; j++) {
    // Copy the column j of B into one of slots in dB using the stream no (j % nStreams) and cudaMemcpyAsync
    // TODO / A FAIRE ...
    int streamId = j % nStreams;

    float *dB_col = dB + streamId*N;
    float *dC_col = dC + streamId*N;

    float *B_col = B + j*N;
    float *C_col = C + j*N;

    cudaMemcpyAsync(dB_col, B_col, N * sizeof(float), cudaMemcpyHostToDevice, streams[streamId]);

    // Perform the matrix-vector multiplication on A and the column vector in dB(:, j % nStreams), compute on dC(:, j % nStreams), using stream no (j % nStreams)
    // TODO / A FAIRE ...
    matvec<<<1, N, 0, streams[streamId] >>>(dA, dB_col, dC_col, N);

    // Copy back the computed vector dC(:, j % nStreams) into the column C(:, j) using the same stream no (j % nStreams) and cudaMemcpyAsync
    cudaMemcpyAsync(C_col, dC_col, N * sizeof(float), cudaMemcpyDeviceToHost, streams[streamId]);
  }
  
  cudaDeviceSynchronize();

  verifyResults();

  free(A); free(B); free(C);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
