#include <stdio.h>
#include <stddef.h>

void bubbleSort(int *arr, size_t n) {
    if (arr == NULL || n < 2) {
        return; // nothing to do or invalid input
    }

    for (size_t i = 0; i + 1 < n; ++i) {
        int swapped = 0;
        // Ensure we never compute (n - i - 1) on unsigned; use j + 1 < n - i
        for (size_t j = 0; j + 1 < n - i; ++j) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = 1;
            }
        }
        if (!swapped) {
            break; // optimization: already sorted
        }
    }
}

int main(void) {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    size_t n = sizeof(arr) / sizeof(arr[0]); // use size_t to avoid truncation

    fputs("Original array: ", stdout);
    for (size_t i = 0; i < n; ++i) {
        printf("%d ", arr[i]);
    }
    putchar('\n');

    bubbleSort(arr, n);

    fputs("Sorted array: ", stdout);
    for (size_t i = 0; i < n; ++i) {
        printf("%d ", arr[i]);
    }
    putchar('\n');

    return 0;
}