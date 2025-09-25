#include <cuda_runtime.h>
#include <stdio.h>
#include <limits.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

__global__ void parallelBinarySearch(int* d_elements, int target, int segment_size, 
                                    int total_elements, bool* d_found, int* d_found_index) {
    // Input validation
    if (!d_elements || !d_found || !d_found_index || segment_size <= 0 || total_elements <= 0) {
        return;
    }
    
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Prevent integer overflow and calculate bounds safely
    long long start_ll = (long long)threadId * segment_size;
    if (start_ll >= total_elements || start_ll > INT_MAX) {
        return; // Thread has no work to do
    }
    
    int start = (int)start_ll;
    int end = min(start + segment_size - 1, total_elements - 1);
    
    // Ensure we don't go out of bounds
    if (start >= total_elements) {
        return;
    }
    
    int low = start, high = end;
    while(low <= high) {
        int mid = low + (high - low) / 2;
        
        // Additional bounds check for safety
        if (mid < 0 || mid >= total_elements) {
            break;
        }
        
        if(d_elements[mid] == target) {
            // Use atomic operation to avoid race condition
            // Only update if not already found (first finder wins)
            bool expected = false;
            bool desired = true;
            
            // Atomic compare and swap - only one thread will succeed
            if (atomicCAS((int*)d_found, (int)expected, (int)desired) == (int)expected) {
                // Store the index of found element atomically
                atomicExch(d_found_index, mid);
            }
            return;
        }
        
        if(d_elements[mid] < target)
            low = mid + 1;
        else
            high = mid - 1;
    }
}

// Helper function to verify array is sorted
__global__ void verifySorted(int* d_elements, int n, bool* d_is_sorted) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1) {
        if (d_elements[idx] > d_elements[idx + 1]) {
            *d_is_sorted = false;
        }
    }
}

int main() {
    const int N = 1000000; // Total number of elements
    const int target = 42;  // Element to search for
    
    // Validate N is positive
    if (N <= 0) {
        fprintf(stderr, "Invalid array size\n");
        return 1;
    }
    
    int* d_elements = nullptr;
    bool* d_found = nullptr;
    int* d_found_index = nullptr;
    bool* d_is_sorted = nullptr;
    
    // Allocate device memory with error checking
    CUDA_CHECK(cudaMalloc(&d_elements, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_found_index, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_sorted, sizeof(bool)));
    
    // Initialize found flag to false and found_index to -1
    bool initial_found = false;
    int initial_index = -1;
    bool initial_sorted = true;
    CUDA_CHECK(cudaMemcpy(d_found, &initial_found, sizeof(bool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_found_index, &initial_index, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_is_sorted, &initial_sorted, sizeof(bool), cudaMemcpyHostToDevice));
    
    // ... (populate d_elements with sorted data)
    
    // Verify array is sorted
    int verifyThreads = 256;
    int verifyBlocks = (N + verifyThreads - 1) / verifyThreads;
    verifySorted<<<verifyBlocks, verifyThreads>>>(d_elements, N, d_is_sorted);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    bool is_sorted;
    CUDA_CHECK(cudaMemcpy(&is_sorted, d_is_sorted, sizeof(bool), cudaMemcpyDeviceToHost));
    
    if (!is_sorted) {
        fprintf(stderr, "Array is not sorted. Binary search requires sorted array.\n");
        cudaFree(d_elements);
        cudaFree(d_found);
        cudaFree(d_found_index);
        cudaFree(d_is_sorted);
        return 1;
    }
    
    // Calculate segment size and grid dimensions safely
    int threadsPerBlock = 256;
    int totalThreads = min(threadsPerBlock * 65535, N); // Limit to max grid size
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    
    // Calculate segment size per thread (ensure at least 1)
    int segment_size = max(1, (N + totalThreads - 1) / totalThreads);
    
    // Launch kernel
    parallelBinarySearch<<<blocksPerGrid, threadsPerBlock>>>(
        d_elements, target, segment_size, N, d_found, d_found_index);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Retrieve results
    bool found;
    int found_index;
    CUDA_CHECK(cudaMemcpy(&found, d_found, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&found_index, d_found_index, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (found) {
        printf("Element %d found at index %d\n", target, found_index);
    } else {
        printf("Element %d not found\n", target);
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_elements));
    CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaFree(d_found_index));
    CUDA_CHECK(cudaFree(d_is_sorted));
    
    return 0;
}