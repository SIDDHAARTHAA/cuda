#include <cstdio>
#include <cuda_runtime.h>

// An empty kernel running on the device
__global__ void empty_kernel() {
    // intentionally empty: demonstrates a device kernel launch
}

int main() {
    std::puts("Launching empty kernel on device...");
    empty_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    std::puts("Kernel finished (control returned to host).");
    return 0;
}