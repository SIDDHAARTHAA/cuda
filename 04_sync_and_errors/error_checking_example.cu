#include <cstdio>
#include <cuda_runtime.h>

// Show how to check errors after CUDA calls
int main() {
    const int N = 10;
    size_t bytes = N * sizeof(int);
    int h[N]; for (int i = 0; i < N; ++i) h[i] = i;

    int *d = nullptr; // intentionally left null to provoke an error
    // This cudaMemcpy will fail because 'd' is not a valid device pointer
    cudaError_t err = cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    } else {
        std::puts("cudaMemcpy succeeded (unexpected)");
    }

    // Always check last error after kernel launches too
    // Example pattern (safe to call even if no kernel):
    cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) std::printf("Last CUDA error: %s\n", cudaGetErrorString(last));

    return 0;
}