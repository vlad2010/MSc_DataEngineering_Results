#include <cuda_runtime.h>
#include <stdio.h>

__global__ void parallelBinarySearch(int* d_elements, int N, int target, int segment_size, bool* d_found) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int start = threadId * segment_size;
    int end = min(start + segment_size - 1, N - 1); // Ensure end is within bounds

    int low = start, high = end;
    while(low <= high) {
        int mid = low + (high - low) / 2;
        if(mid < 0 || mid >= N) // Bounds check
            break;
        if(d_elements[mid] == target) {
            // Atomically set d_found to true
            atomicExch(d_found, true);
            return;
        }
        if(d_elements[mid] < target)
            low = mid + 1;
        else
            high = mid - 1;
    }
}

int main() {
    int N = /* ... set array size ... */;
    int target = /* ... set target value ... */;
    int segment_size = /* ... set segment size ... */;
    int* d_elements; // device pointer to the array of elements
    bool* d_found; // device pointer to a boolean variable indicating whether the element is found

    // ... (allocate and copy d_elements as needed) ...

    // Allocate and initialize d_found to false
    cudaMalloc(&d_found, sizeof(bool));
    bool h_found = false;
    cudaMemcpy(d_found, &h_found, sizeof(bool), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    parallelBinarySearch<<<blocksPerGrid, threadsPerBlock>>>(d_elements, N, target, segment_size, d_found);

    // Retrieve result
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    // ... (clean up memory) ...
    cudaFree(d_found);

    // ... (rest of your code) ...
}