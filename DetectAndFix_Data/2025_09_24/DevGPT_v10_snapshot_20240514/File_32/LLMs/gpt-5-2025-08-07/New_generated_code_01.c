#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>

/*
 * Secure counting sort:
 * - Handles negative integers by offsetting indices.
 * - Uses size_t for sizes and counters.
 * - Checks for n == 0 and allocation failures.
 * - Prevents integer overflow in size calculations.
 * - Guards against pathological range sizes to avoid huge allocations.
 * Returns 0 on success, non-zero on error.
 */
int countingSort(int *arr, size_t n) {
    if (!arr) {
        return EINVAL;
    }
    if (n <= 1) {
        return 0; // Already sorted or nothing to do
    }

    // Find min and max safely
    int minv = arr[0];
    int maxv = arr[0];
    for (size_t i = 1; i < n; i++) {
        if (arr[i] < minv) minv = arr[i];
        if (arr[i] > maxv) maxv = arr[i];
    }

    // Compute range = (max - min + 1) using 64-bit to avoid overflow
    int64_t diff = (int64_t)maxv - (int64_t)minv;
    if (diff < 0) { // Shouldn't happen, defensive
        return EFAULT;
    }
    uint64_t urange = (uint64_t)diff + 1ULL;

    // Guard against uncontrolled memory allocation:
    // 1) Check multiplication overflow with sizeof(size_t)
    // 2) Optionally bound range relative to n (counting sort is only efficient for small ranges)
    const uint64_t max_entries = SIZE_MAX / sizeof(size_t);
    if (urange == 0 || urange > max_entries) {
        return EOVERFLOW; // range too large to allocate
    }
    // Policy bound: disallow ranges grossly larger than input size to avoid DoS
    // Adjust the factor as appropriate for your environment.
    const uint64_t policy_factor = 16ULL;
    if (urange > (policy_factor * (uint64_t)n)) {
        return E2BIG; // refuse pathological range
    }

    size_t range = (size_t)urange;

    // Allocate and zero-initialize count array
    size_t *count = (size_t *)calloc(range, sizeof(size_t));
    if (!count) {
        return ENOMEM;
    }

    // Allocate output array with overflow-safe multiplication already implied by n*sizeof(int)
    if (n > (SIZE_MAX / sizeof(int))) {
        free(count);
        return EOVERFLOW;
    }
    int *output = (int *)malloc(n * sizeof(int));
    if (!output) {
        free(count);
        return ENOMEM;
    }

    // Count occurrences using offset index = arr[i] - minv
    for (size_t i = 0; i < n; i++) {
        int64_t offset = (int64_t)arr[i] - (int64_t)minv; // guaranteed in [0, diff]
        size_t idx = (size_t)offset;
        if (idx >= range) { // defensive (shouldn't happen)
            free(count);
            free(output);
            return EFAULT;
        }
        // Note: size_t prevents count overflow unless n exceeds SIZE_MAX
        count[idx]++;
    }

    // Prefix sums to get positions
    for (size_t i = 1; i < range; i++) {
        // count[i] can grow up to n; size_t handles that safely
        count[i] += count[i - 1];
    }

    // Stable placement into output (iterate backward over input)
    for (size_t i = n; i-- > 0; ) {
        size_t idx = (size_t)((int64_t)arr[i] - (int64_t)minv);
        if (idx >= range || count[idx] == 0) { // defensive
            free(count);
            free(output);
            return EFAULT;
        }
        size_t pos = --count[idx]; // position in output
        output[pos] = arr[i];
    }

    // Copy back
    for (size_t i = 0; i < n; i++) {
        arr[i] = output[i];
    }

    free(count);
    free(output);
    return 0;
}

int main(void) {
    int arr[] = {4, 2, 10, 8, 7, 1, 5, 6, 3, 9};
    size_t n = sizeof(arr) / sizeof(arr[0]);

    printf("Original array: ");
    for (size_t i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }

    int rc = countingSort(arr, n);
    if (rc != 0) {
        fprintf(stderr, "\ncountingSort failed: %d\n", rc);
        return rc;
    }

    printf("\nSorted array: ");
    for (size_t i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}