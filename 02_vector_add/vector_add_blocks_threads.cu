#include <cstdio>
#include <cuda_runtime.h>

// Vector add: many blocks and many threads (general form)
__global__ void vecAdd(const float* a, const float* b, float* c, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // global index
    if (i < N) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1000;
    size_t bytes = N * sizeof(float);
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) { h_a[i] = i * 1.0f; h_b[i] = i * 0.5f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    std::puts("Sample results (general vector add):");
    for (int i = 0; i < 5; ++i) std::printf("%d: %f + %f = %f\n", i, h_a[i], h_b[i], h_c[i]);

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}