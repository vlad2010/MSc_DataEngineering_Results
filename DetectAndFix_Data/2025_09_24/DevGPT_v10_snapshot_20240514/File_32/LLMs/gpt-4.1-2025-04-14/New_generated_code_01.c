#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int isValidInput(const int arr[], int n, int min, int max) {
    for (int i = 0; i < n; i++) {
        if (arr[i] < min || arr[i] > max) {
            return 0;
        }
    }
    return 1;
}

void countingSort(int arr[], int n) {
    if (arr == NULL || n <= 0) {
        fprintf(stderr, "Invalid input: array is NULL or size is non-positive.\n");
        return;
    }

    // Find the maximum and minimum element to determine the range
    int max = arr[0];
    int min = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) max = arr[i];
        if (arr[i] < min) min = arr[i];
    }

    if (min < 0) {
        fprintf(stderr, "Error: Negative numbers not supported in this counting sort implementation.\n");
        return;
    }

    // Prevent integer overflow in malloc
    if (max > INT_MAX - 1) {
        fprintf(stderr, "Error: max value too large.\n");
        return;
    }
    size_t count_size = (size_t)max + 1;
    if (count_size > SIZE_MAX / sizeof(int)) {
        fprintf(stderr, "Error: count array size too large.\n");
        return;
    }

    int* count = (int*)malloc(count_size * sizeof(int));
    if (count == NULL) {
        fprintf(stderr, "Memory allocation failed for count array.\n");
        return;
    }

    // Initialize the counting array
    for (size_t i = 0; i < count_size; i++) {
        count[i] = 0;
    }

    // Validate input values are within [0, max]
    if (!isValidInput(arr, n, 0, max)) {
        fprintf(stderr, "Error: Input array contains out-of-range values.\n");
        free(count);
        return;
    }

    // Count the occurrences of each element in the input array
    for (int i = 0; i < n; i++) {
        count[arr[i]]++;
    }

    // Modify the counting array to store the position of each element
    for (int i = 1; i <= max; i++) {
        count[i] += count[i - 1];
    }

    int* output = (int*)malloc(n * sizeof(int));
    if (output == NULL) {
        fprintf(stderr, "Memory allocation failed for output array.\n");
        free(count);
        return;
    }

    // Build the output array by placing elements in their correct positions
    for (int i = n - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }

    // Copy the sorted elements back to the original array
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }

    free(count);
    free(output);
}

int main() {
    int arr[] = {4, 2, 10, 8, 7, 1, 5, 6, 3, 9};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Original array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }

    countingSort(arr, n);

    printf("\nSorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }

    return 0;
}