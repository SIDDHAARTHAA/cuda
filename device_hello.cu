#include <stdio.h>

__global__ void mykernel() {
    // runs on GPU
}

int main() {
    mykernel<<<1,1>>>();   // launch GPU kernel
    printf("Hello World from CPU!\n");
    return 0;
}
    