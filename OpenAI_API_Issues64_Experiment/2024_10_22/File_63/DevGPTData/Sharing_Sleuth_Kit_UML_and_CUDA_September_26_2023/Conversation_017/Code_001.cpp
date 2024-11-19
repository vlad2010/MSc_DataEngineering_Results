__global__ void parallelBinarySearch(int* d_array, int target, bool* d_found, int* d_direction, int segment_size) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int start = threadId * segment_size;
    int end = start + segment_size - 1;
    
    // Each thread performs binary search in its segment
    int low = start, high = end;
    while(low <= high) {
        int mid = low + (high - low) / 2;
        if(d_array[mid] == target) {
            *d_found = true;
            return;
        }
        if(d_array[mid] < target)
            low = mid + 1;
        else
            high = mid - 1;
    }
    
    // Report back the direction where the element could be
    d_direction[threadId] = (d_array[start] < target) ? 1 : -1;
}
