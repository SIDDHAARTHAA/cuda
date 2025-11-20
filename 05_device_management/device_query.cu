#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    std::printf("Device count: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::printf("Device %d: %s \n  SMs: %d, Compute capability: %d.%d, totalGlobalMem: %zu MB\n",
                    i, prop.name, prop.multiProcessorCount, prop.major, prop.minor, prop.totalGlobalMem / (1024*1024));
    }
    return 0;
}