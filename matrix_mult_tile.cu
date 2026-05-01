#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32

__global__ void matrix_multiplication_kernel(float *a, float *b, float *c, int N) {

  __shared__ float a_s[TILE_DIM][TILE_DIM];
  __shared__ float b_s[TILE_DIM][TILE_DIM];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0;

  for (int tile = 0; tile < (N + TILE_DIM - 1) / TILE_DIM; tile++) {

      if (row < N && (tile * TILE_DIM + threadIdx.x) < N)
        a_s[threadIdx.y][threadIdx.x] = a[row * N + tile * TILE_DIM + threadIdx.x];
      else
        a_s[threadIdx.y][threadIdx.x] = 0.0;

      if (col < N && (tile * TILE_DIM + threadIdx.y) < N)
        b_s[threadIdx.y][threadIdx.x] = b[(tile * TILE_DIM + threadIdx.y) * N + col];
      else
        b_s[threadIdx.y][threadIdx.x] = 0.0;

      __syncthreads();

      for (int k = 0; k < TILE_DIM; k++) {
        sum += a_s[threadIdx.y][k] * b_s[k][threadIdx.x];
      }
      __syncthreads();
  }

    if(row < N && col < N){
        c[row * N + col] = sum;
    }

}

void marshalling(float *h_a, float *h_b, float *h_c, int N) {
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  cudaMalloc(&d_a, N * N * sizeof(float));
  cudaMalloc(&d_b, N * N * sizeof(float));
  cudaMalloc(&d_c, N * N * sizeof(float));

  cudaMemcpy(d_a, h_a, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, N * N * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = TILE_DIM;
  dim3 threadsPerBlock(blockSize, blockSize);
  dim3 blocksPerGrid((N + blockSize - 1) / blockSize,(N + blockSize - 1) / blockSize);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaMemcpy(h_c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("%f", milliseconds);

/*
  for (int i = 0; i < N; i += 400) {
    for (int j = 0; j < N; j += 400) {
      printf("%.2f ", h_c[i * N + j]);
    }
    printf("\n");
  }
*/

}

int main(int argc, char **argv) {

  if (argc != 2) {
    printf("Usage: %s <matrix_size>\n", argv[0]);
    return -1;
  }

  int N = atoi(argv[1]);

  float *mat_a, *mat_b, *mat_c;

  // Allocating memory for the matrices
  mat_a = (float *)malloc(N * N * sizeof(float));
  mat_b = (float *)malloc(N * N * sizeof(float));
  mat_c = (float *)malloc(N * N * sizeof(float));

  // Initializing the matrices
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      mat_a[i * N + j] = 1.0;
      mat_b[i * N + j] = 1.0;

      // Need to initialize matrix c to 0, if we don't, garbage values get added to result
      mat_c[i * N + j] = 0.0;
    }
  }

  marshalling(&mat_a[0], &mat_b[0], &mat_c[0], N);

  free(mat_a);
  free(mat_b);
  free(mat_c);

  return 0;
}
