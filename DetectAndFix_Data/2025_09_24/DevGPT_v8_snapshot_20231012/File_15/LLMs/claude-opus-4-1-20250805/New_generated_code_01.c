#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

// Return value indicates success/failure, parameters include output buffer capacity
bool findIntersection(const int arr1[], int size1, 
                     const int arr2[], int size2, 
                     int intersection[], int capacity, int *size3);

int main() {
    int arr1[] = {1, 2, 3, 4, 5};
    int size1 = sizeof(arr1) / sizeof(arr1[0]);

    int arr2[] = {3, 4, 5, 6, 7};
    int size2 = sizeof(arr2) / sizeof(arr2[0]);

    const int INTERSECTION_CAPACITY = 10;
    int intersection[INTERSECTION_CAPACITY];
    int size3 = 0;  // Initialize to prevent undefined behavior

    bool success = findIntersection(arr1, size1, arr2, size2, 
                                   intersection, INTERSECTION_CAPACITY, &size3);

    if (success) {
        printf("Intersection (size=%d): ", size3);
        for (int i = 0; i < size3; i++) {
            printf("%d ", intersection[i]);
        }
        printf("\n");
    } else {
        fprintf(stderr, "Error: Failed to compute intersection\n");
    }

    return 0;
}

bool findIntersection(const int arr1[], int size1, 
                     const int arr2[], int size2, 
                     int intersection[], int capacity, int *size3) {
    // Input validation
    if (arr1 == NULL || arr2 == NULL || intersection == NULL || size3 == NULL) {
        return false;
    }
    
    if (size1 < 0 || size2 < 0 || capacity <= 0) {
        return false;
    }
    
    int count = 0;
    
    // Find intersection with bounds checking
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            if (arr1[i] == arr2[j]) {
                // Check for duplicates in result
                bool already_added = false;
                for (int k = 0; k < count; k++) {
                    if (intersection[k] == arr1[i]) {
                        already_added = true;
                        break;
                    }
                }
                
                if (!already_added) {
                    // Bounds check before adding to intersection
                    if (count >= capacity) {
                        // Buffer would overflow, return what we have so far
                        *size3 = count;
                        return false;  // Indicate partial result
                    }
                    intersection[count++] = arr1[i];
                }
                break;  // Element found, no need to continue inner loop
            }
        }
    }
    
    *size3 = count;
    return true;
}