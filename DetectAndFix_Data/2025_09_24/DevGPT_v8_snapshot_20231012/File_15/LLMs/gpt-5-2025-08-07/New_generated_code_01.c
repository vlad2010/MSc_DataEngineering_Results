#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

void findIntersection(const int *arr1, size_t size1,
                      const int *arr2, size_t size2,
                      int *intersection, size_t intersectionCapacity,
                      size_t *size3);

int main(void) {
    int arr1[] = {1, 2, 3, 4, 5};
    size_t size1 = sizeof(arr1) / sizeof(arr1[0]);

    int arr2[] = {3, 4, 5, 6, 7};
    size_t size2 = sizeof(arr2) / sizeof(arr2[0]);

    // Capacity is at most the smaller of the two arrays
    size_t capacity = (size1 < size2) ? size1 : size2;

    // Allocate intersection buffer sized to the true maximum
    int *intersection = NULL;
    if (capacity > 0) {
        intersection = (int *)malloc(capacity * sizeof(*intersection));
        if (intersection == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
    }

    size_t size3 = 0; // Initialize to prevent use-before-set
    findIntersection(arr1, size1, arr2, size2, intersection, capacity, &size3);

    printf("Intersection: ");
    for (size_t i = 0; i < size3; i++) {
        printf("%d ", intersection[i]);
    }
    printf("\n");

    free(intersection);
    return 0;
}

void findIntersection(const int *arr1, size_t size1,
                      const int *arr2, size_t size2,
                      int *intersection, size_t intersectionCapacity,
                      size_t *size3) {
    // Validate required pointers
    if (size3 == NULL) {
        return; // can't report size; nothing safe to do
    }
    *size3 = 0; // Always set to a known value

    if (arr1 == NULL || arr2 == NULL) {
        return;
    }
    if (intersectionCapacity == 0) {
        return; // nothing to write; size3 already set to 0
    }
    if (intersection == NULL) {
        return; // capacity > 0 but no buffer
    }

    // Compute intersection without duplicates
    for (size_t i = 0; i < size1; i++) {
        int val = arr1[i];

        // Check if val exists in arr2
        bool in_arr2 = false;
        for (size_t j = 0; j < size2; j++) {
            if (val == arr2[j]) {
                in_arr2 = true;
                break;
            }
        }
        if (!in_arr2) {
            continue;
        }

        // Avoid duplicates in the intersection result
        bool already_added = false;
        for (size_t k = 0; k < *size3; k++) {
            if (intersection[k] == val) {
                already_added = true;
                break;
            }
        }
        if (already_added) {
            continue;
        }

        // Enforce capacity to prevent buffer overflow (CWE-120/CWE-119)
        if (*size3 < intersectionCapacity) {
            intersection[*size3] = val;
            (*size3)++;
        } else {
            // Capacity reached; stop to avoid overflow
            break;
        }
    }
}