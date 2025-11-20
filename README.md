# CUDA Programming Tutorial

This repository contains simple CUDA C++ examples to help you get started with GPU programming using NVIDIA's CUDA platform. CUDA allows you to write programs that execute on NVIDIA GPUs, enabling massive parallelism for computationally intensive tasks.

## What is CUDA?
CUDA (Compute Unified Device Architecture) is a parallel computing platform and API model created by NVIDIA. It enables developers to use C, C++, and Fortran to write programs that run on the GPU, leveraging thousands of cores for high-performance computing.

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed ([Download here](https://developer.nvidia.com/cuda-downloads))
- C++ compiler (e.g., MSVC, GCC, Clang)

## File Overview
- `hello.cu`: Basic "Hello, World!" example using CUDA
- `device_hello.cu`: Demonstrates device-side printing
- `add_two.cu`: Adds two numbers using CUDA kernel
- `add_two/`: (Optional) Directory for extended examples

## How CUDA Works
1. **Host vs Device**: The CPU is called the "host" and the GPU is the "device".
2. **Kernels**: Functions executed on the GPU are called kernels. They are launched from the host.
3. **Memory Management**: Data must be transferred between host and device memory using CUDA API functions.
4. **Parallel Execution**: Kernels are executed by many threads in parallel.

## Basic Syntax

### 1. Kernel Definition
```cpp
__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}
```
- `__global__` indicates a function that runs on the device and is called from the host.

### 2. Kernel Launch
```cpp
add<<<1, 1>>>(dev_a, dev_b, dev_c);
```
- `<<<1, 1>>>` specifies the grid and block dimensions (here, 1 block of 1 thread).

### 3. Memory Allocation
```cpp
int *dev_a;
cudaMalloc((void**)&dev_a, sizeof(int));
```
- `cudaMalloc` allocates memory on the device.

### 4. Memory Copy
```cpp
cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);
```
- `cudaMemcpy` copies data between host and device.

### 5. Synchronization
```cpp
cudaDeviceSynchronize();
```
- Ensures all device operations are complete before proceeding.

## Example: Adding Two Numbers
See `add_two.cu` for a complete example.

## Compiling and Running
1. Open a terminal in this directory.
2. Compile a CUDA file using `nvcc`:
   ```powershell
   nvcc add_two.cu -o add_two.exe
   ```
3. Run the executable:
   ```powershell
   .\add_two.exe
   ```

## Resources
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Developer Zone](https://developer.nvidia.com/)

## Notes
- Always check for errors after CUDA API calls using `cudaGetLastError()`.
- Use `nvcc` to compile `.cu` files.
- Grid and block dimensions control parallelism.

---
Feel free to explore the example files and modify them to learn more about CUDA programming!