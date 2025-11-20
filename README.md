# CUDA Tutorial

A compact, step-by-step CUDA learning codebase. Each folder focuses on one core concept and includes short, clear `.cu` examples that compile and run with `nvcc`.

**Folder structure**

```
cuda-tutorial/
    README.md
    01_hello_world/
        hello_host.cu
        hello_device.cu
    02_vector_add/
        vector_add_blocks.cu
        vector_add_threads.cu
        vector_add_blocks_threads.cu
    03_shared_memory_stencil/
        stencil_basic.cu
        stencil_shared_memory.cu
    04_sync_and_errors/
        sync_example.cu
        error_checking_example.cu
    05_device_management/
        device_query.cu
```

**Quick concepts**

- **Host vs Device:** Host = CPU; Device = GPU. Host code runs on CPU and launches kernels that run on the device.
- **Kernel launch basics:** `kernel<<<gridDim, blockDim>>>(...)` launches work on the GPU.
- **Blocks and threads:** Threads are grouped into blocks; blocks form a grid. Each thread has `threadIdx`, each block has `blockIdx`.
- **Global index formula:** `int idx = threadIdx.x + blockIdx.x * blockDim.x;` maps a thread to a unique global index.
- **Memory management:** Use `cudaMalloc`, `cudaMemcpy`, and `cudaFree` to manage device memory.
- **Shared memory basics:** Use `extern __shared__` or `__shared__` arrays for fast block-local memory.
- **Halo concept for stencil:** A halo (ghost) region holds neighbor elements so threads can compute stencils without redundant global loads.
- **Using `__syncthreads()`:** Synchronizes threads within a block to ensure shared memory writes are visible.
- **Error checking:** Use `cudaGetLastError()` and `cudaGetErrorString()` to inspect runtime errors.
- **Device management:** Use `cudaGetDeviceCount()` and `cudaGetDeviceProperties()` to query GPUs.

**Build & run (WSL / Linux)**

Compile:

```bash
nvcc file.cu -o file
```

Run:

```bash
./file
```

**Notes**
- All examples are intentionally small to emphasize the single concept they teach.
- Use `nvcc --help` or `nvcc --version` to verify your CUDA toolchain.
- If a program fails, check the output of `cudaGetLastError()` (see `04_sync_and_errors/error_checking_example.cu`).

Explore the folders to study each concept and run examples one by one.