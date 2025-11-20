#include <cstdio>
#include <cuda_runtime.h>

// Simple 1D stencil (radius=1) reading only from global memory
__global__ void stencil_basic(const float* in, float* out, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;

    float left = (i > 0) ? in[i-1] : in[i];
    float center = in[i];
    float right = (i < N-1) ? in[i+1] : in[i];

    out[i] = (left + center + right) / 3.0f;
}

int main() {
    const int N = 16;
    size_t bytes = N * sizeof(float);

    float h_in[N], h_out[N];
    for (int i = 0; i < N; ++i) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int blockSize = 8;
    int gridSize = (N + blockSize - 1) / blockSize;
    stencil_basic<<<gridSize, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    std::puts("Stencil (global loads only):");
    for (int i = 0; i < N; ++i) std::printf("%d -> %f\n", i, h_out[i]);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}