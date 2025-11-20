#include <cstdio>
#include <cuda_runtime.h>

// 1D stencil using shared memory and halo elements (radius = 1)
__global__ void stencil_shared(const float* in, float* out, int N) {
    extern __shared__ float s[]; // shared mem size = blockDim.x + 2
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;

    // Load center
    s[t+1] = (i < N) ? in[i] : 0.0f;
    // Load halos
    if (t == 0) {
        s[0] = (i > 0) ? in[i-1] : s[1];
    }
    if (t == blockDim.x - 1) {
        int rightIdx = i + 1;
        s[blockDim.x + 1] = (rightIdx < N) ? in[rightIdx] : s[blockDim.x];
    }

    __syncthreads();

    if (i < N) {
        float left = s[t];
        float center = s[t+1];
        float right = s[t+2];
        out[i] = (left + center + right) / 3.0f;
    }
}

int main() {
    const int N = 64;
    size_t bytes = N * sizeof(float);
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int blockSize = 16;
    int gridSize = (N + blockSize - 1) / blockSize;
    size_t sharedBytes = (blockSize + 2) * sizeof(float);

    stencil_shared<<<gridSize, blockSize, sharedBytes>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    std::puts("Stencil using shared memory (sample):");
    for (int i = 0; i < 8; ++i) std::printf("%d -> %f\n", i, h_out[i]);

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}