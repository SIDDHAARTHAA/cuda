#include <cstdio>
#include <cuda_runtime.h>

// Demonstrate need for __syncthreads(): two kernels, without and with sync
__global__ void without_sync(int *out) {
    __shared__ int s[2];
    int t = threadIdx.x;
    // thread 1 writes after a delay loop
    if (t == 1) {
        for (volatile int i = 0; i < 1000; ++i); // small delay
        s[t] = 42;
    }
    // thread 0 reads without sync
    if (t == 0) {
        out[0] = s[1]; // may read uninitialized because no sync
    }
}

__global__ void with_sync(int *out) {
    __shared__ int s[2];
    int t = threadIdx.x;
    if (t == 1) {
        for (volatile int i = 0; i < 1000; ++i);
        s[t] = 42;
    }
    __syncthreads(); // ensure writes are visible
    if (t == 0) {
        out[0] = s[1]; // now should reliably see 42
    }
}

int main() {
    int h_out[1] = {0};
    int *d_out;
    cudaMalloc((void**)&d_out, sizeof(int));

    // Launch with 1 block, 2 threads
    without_sync<<<1,2>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    std::printf("Without sync read: %d (undefined/likely 0)\n", h_out[0]);

    with_sync<<<1,2>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    std::printf("With sync read: %d (should be 42)\n", h_out[0]);

    cudaFree(d_out);
    return 0;
}