#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _err = (expr);                                           \
        if (_err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                  \
                    #expr, __FILE__, __LINE__, cudaGetErrorString(_err));    \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

__global__ void parallelBinarySearch(
    const int* __restrict__ d_elements,
    size_t N,
    int target,
    size_t segment_size,
    int* __restrict__ d_found // 0 = not found, 1 = found
) {
    // Defensive: handle invalid inputs quickly
    if (segment_size == 0 || N == 0) return;

    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t numSegments = (N + segment_size - 1) / segment_size;
    if (tid >= numSegments) return;

    // If someone already found it, exit early
    if (atomicAdd(d_found, 0) != 0) return;

    size_t start = tid * segment_size;
    size_t end_inclusive = start + segment_size;
    if (end_inclusive > N) end_inclusive = N;
    if (start >= end_inclusive) return;
    size_t low = start;
    size_t high = end_inclusive - 1;

    while (low <= high) {
        // Check again to reduce unnecessary work
        if (atomicAdd(d_found, 0) != 0) return;

        size_t mid = low + ((high - low) >> 1); // avoids overflow
        int val = d_elements[mid];

        if (val == target) {
            atomicExch(d_found, 1);
            return;
        } else if (val < target) {
            low = mid + 1;
        } else {
            if (mid == 0) break; // prevent size_t underflow
            high = mid - 1;
        }
    }
}

int main() {
    // Example parameters (ensure input is sorted for binary search)
    const size_t N = 1 << 20; // 1,048,576 elements
    const size_t segment_size = 1024; // must be > 0
    const int target = 123456;

    // Allocate and initialize host data (sorted for binary search)
    int* h_elements = nullptr;
    try {
        h_elements = new int[N];
    } catch (...) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }
    for (size_t i = 0; i < N; ++i) {
        h_elements[i] = static_cast<int>(i); // strictly increasing
    }

    // Device allocations
    int* d_elements = nullptr;
    int* d_found = nullptr;
    CUDA_CHECK(cudaMalloc(&d_elements, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));

    // Copy data to device and initialize found flag to 0
    CUDA_CHECK(cudaMemcpy(d_elements, h_elements, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found, 0, sizeof(int)));

    // Compute launch configuration based on number of segments (not N)
    const int threadsPerBlock = 256;
    if (segment_size == 0) {
        fprintf(stderr, "segment_size must be > 0\n");
        // Cleanup and exit
        CUDA_CHECK(cudaFree(d_elements));
        CUDA_CHECK(cudaFree(d_found));
        delete[] h_elements;
        return 1;
    }
    size_t numSegments = (N + segment_size - 1) / segment_size;
    int blocksPerGrid = static_cast<int>((numSegments + threadsPerBlock - 1) / threadsPerBlock);

    // Launch kernel
    parallelBinarySearch<<<blocksPerGrid, threadsPerBlock>>>(d_elements, N, target, segment_size, d_found);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Retrieve result
    int h_found = 0;
    CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost));

    printf("Found? %s\n", (h_found != 0 ? "yes" : "no"));

    // Cleanup
    CUDA_CHECK(cudaFree(d_elements));
    CUDA_CHECK(cudaFree(d_found));
    delete[] h_elements;

    return 0;
}