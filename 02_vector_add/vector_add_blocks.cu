#include <cstdio>
#include <cuda_runtime.h>

// Vector add: N blocks, 1 thread per block
__global__ void vecAdd_blocks(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x; // one thread per block
    if (i < N) c[i] = a[i] + b[i];
}

int main() {
    const int N = 8;
    size_t bytes = N * sizeof(float);
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; ++i) { h_a[i] = i; h_b[i] = i * 2; }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch N blocks, 1 thread per block
    vecAdd_blocks<<<N, 1>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    std::puts("Results (N blocks, 1 thread each):");
    for (int i = 0; i < N; ++i) std::printf("%d: %f + %f = %f\n", i, h_a[i], h_b[i], h_c[i]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}