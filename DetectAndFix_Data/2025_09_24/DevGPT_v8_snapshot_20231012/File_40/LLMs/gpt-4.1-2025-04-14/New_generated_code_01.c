#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define MAX_ARRAY_SIZE 100

void bubbleSort(int arr[], int n) {
    if (arr == NULL || n < 0 || n > MAX_ARRAY_SIZE) { // CWE-119: Bounds check
        fprintf(stderr, "Invalid array or size\n");
        return;
    }
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(arr) / sizeof(arr[0]);

    if (n < 0 || n > MAX_ARRAY_SIZE) { // CWE-20: Input validation
        fprintf(stderr, "Array size out of bounds\n");
        return 1;
    }

    printf("Original array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    bubbleSort(arr, n);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}