# Matrix Multiplication: Tiling with Shared Memory

An optimized implementation of 2D matrix multiplication using **Shared Memory Tiling** in NVIDIA CUDA. This repository marks the next phase of performance tuning, focusing on overcoming global memory bandwidth bottlenecks by caching data close to the execution cores.

## 🚀 The Result

* **Memory-Bound to Compute-Bound:** By leveraging on-chip Shared Memory as a software managed cache, threads eliminate redundant, high latency roundtrips to Global Memory.
* **Scale Capability:** Dropped the sequential CPU baseline entirely because it stalls indefinitely at this scale. This optimized implementation handles massive scales, benchmarked at **16,384 × 16,384** matrix dimensions on a single commodity GPU.

---

## 🧠 Tiling Mechanics: Breaking the Bottleneck

In the naive GPU implementation, every single thread has to read its corresponding row of Matrix A and column of Matrix B directly from off-chip **Global Memory**. This creates an instruction stall where execution units spend more time waiting for memory bus data arrivals than actually executing math operations.

### The Solution: Collaborative Loading
This project implements **2D Tiling**:
1. Threads within a 2D Thread Block (16 × 16) cooperate to load a small sub-matrix tile from Global Memory into fast, on-chip **Shared Memory**.
2. A barrier synchronization (`__syncthreads()`) ensures all threads finish loading the tile.
3. Threads perform a dot-product on the local tile out of high-speed SRAM registers.
4. The block steps forward to the next tile across the matrix width, reusing the fast data layout.

---

## 🛠️ Tech Stack & Profiling Environment

* **Language:** C / C++ (CUDA C)
* **Parallel Computing Architecture:** NVIDIA CUDA 13.0
* **Compiler:** `nvcc` (Target Architecture: `sm_75`)
* **Hardware Tested:** NVIDIA Tesla T4 GPU (15GB VRAM)
* **Profiling Tool:** `nvprof` (NVIDIA Command-line Profiler)

---

## 📊 Profiling & Performance Metrics

Running the tiled kernel at scale on a massive **16,384 × 16,384** matrix workspace yields the following exact resource metrics via `nvprof`:

| Operation / Activity | Time (%) | Total Execution Time | Calls |
| :--- | :--- | :--- | :--- |
| `matrix_multiplication_kernel` | **92.54%** | 12.4167 seconds | 1 |
| Host-to-Device Memory Copy (`memcpy HtoD`) | **5.51%** | 738.81 ms | 3 |
| Device-to-Host Memory Copy (`memcpy DtoH`) | **1.96%** | 262.69 ms | 1 |

### Key Observations
* **High Kernel Saturation:** The actual execution kernel owns **92.54%** of total GPU activity time. This proves that memory marshalling overhead (`cudaMemcpy`) is successfully minimized, allowing the execution hardware to spend its time executing compute cycles rather than stalling on initialization transfers.

---

## 💻 Code Structure & Configuration

The kernel defines a tile dimension footprint matching standard warp boundary mechanics:

```c
#define TILE_DIM 16
```

### Grid and Block Layout Setup

```C
int blockSize = TILE_DIM;
dim3 threadsPerBlock(blockSize, blockSize);
dim3 blocksPerGrid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
```

## 📈 Key System Learnings
 - Barrier Synchronization Check: Solidified the strategic placement of __syncthreads(). It is required not only after loading data into Shared Memory (to prevent a thread from calculating using uninitialized garbage values) but also after the computing loop before moving to the next tile iteration (to prevent fast threads from overwriting shared tile memory while slower threads are still computing their current tile sum).
 - Hardware Saturation Boundaries: Discovered that optimizing for memory latency shifts the focus entirely to execution limitations. While tiling handles 16,384 matrices beautifully, tracking performance changes at this volume sets the foundation for more advanced scaling optimizations like thread coarsening.
