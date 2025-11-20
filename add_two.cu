#include <stdio.h>

// This function runs on the GPU
__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

//device = gpu
//host = cpu

int main() {
    int a = 2, b = 7;      // host values
    int c;                 // result on host

    int *d_a, *d_b, *d_c;  // device pointers

    // Allocate GPU memory
    cudaMalloc((void**)&d_a, sizeof(int));
    cudaMalloc((void**)&d_b, sizeof(int));
    cudaMalloc((void**)&d_c, sizeof(int));

    // Copy inputs to GPU
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    // Launch GPU kernel with 1 block, 1 thread
    add<<<1,1>>>(d_a, d_b, d_c);

    // Copy result back to CPU
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Result: %d\n", c);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
