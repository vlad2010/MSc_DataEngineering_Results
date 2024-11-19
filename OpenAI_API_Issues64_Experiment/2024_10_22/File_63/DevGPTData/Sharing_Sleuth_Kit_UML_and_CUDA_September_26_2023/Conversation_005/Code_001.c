__global__ void parallel_binary_search(int *d_array, int *d_keys, int *d_results, int array_size, int num_keys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_keys) {
        int key = d_keys[tid];
        int left = 0, right = array_size - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (d_array[mid] == key) {
                d_results[tid] = mid; // key found
                return;
            }
            if (d_array[mid] < key)
                left = mid + 1;
            else
                right = mid - 1;
        }
        d_results[tid] = -1; // key not found
    }
}
