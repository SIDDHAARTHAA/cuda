# CUDA Tutorial

This repo teaches CUDA step by step with small `.cu` files you can compile and run.
Each example focuses on one idea so you don’t get lost in big code.

---

## What you need

* A system with an NVIDIA GPU
* CUDA toolkit installed
* `nvcc` in PATH
* Linux or WSL is fine

Check your setup:

```bash
nvcc --version
nvidia-smi
```

---

## How to build and run

Compile any file:

```bash
nvcc file.cu -o file
```

Run it:

```bash
./file
```

If it fails, check the last CUDA error:

```cpp
cudaGetLastError();
cudaGetErrorString(err);
```

---

# Concepts you will learn

## 1. Host and device

* Host is CPU.
* Device is GPU.
* Each has separate memory.
* Host code launches GPU kernels.

## 2. Kernels

A kernel is a GPU function marked with:

```cpp
__global__
```

You launch it from the CPU with:

```cpp
kernel<<<blocks, threads>>>();
```

The CPU does not wait for this by default.

## 3. Blocks and threads

* Threads live inside blocks
* Blocks form the grid
* GPU gives you:

  * `blockIdx.x`
  * `threadIdx.x`
  * `blockDim.x`

## 4. Global index

Every thread needs a unique job.
Use this formula:

```cpp
int idx = threadIdx.x + blockIdx.x * blockDim.x;
```

## 5. Device memory

You must move data to the GPU:

```cpp
cudaMalloc(&d_ptr, bytes);
cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice);
```

Then move results back.

Free when done:

```cpp
cudaFree(d_ptr);
```

## 6. Vector add examples

You will see:

* N blocks, 1 thread each
* 1 block, N threads
* Blocks and threads together (the right way)

## 7. Stencils

A stencil uses neighbors:

```
output[i] = sum of input[i - R] to input[i + R]
```

This needs many reads, so it benefits from shared memory.

## 8. Shared memory

Shared memory is a fast space used by threads in the same block.

Example:

```cpp
__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
```

Threads load:

* their own values
* halo values
  Then they sync and compute.

## 9. Barriers

`__syncthreads()` forces all threads in the block to wait.
Needed so nobody reads shared memory before it is ready.

## 10. Error handling

You can check what went wrong after a kernel launch:

```cpp
cudaGetLastError();
cudaGetErrorString(code);
```

## 11. Device info

You can query the GPU:

```cpp
cudaGetDeviceCount(&n);
cudaGetDeviceProperties(&prop, i);
```

---

# How to use this repo

Go through folders in order:

1. Hello world
2. Vector add
3. Stencil basic
4. Stencil with shared memory
5. Sync and error checking
6. Device info

Run each file.
Read the comments.
Change values and see what happens.

This will give you a solid start and real confidence.

---
