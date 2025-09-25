#include <stdio.h>

// Pass intersection buffer size as an argument
void findIntersection(int arr1[], int size1, int arr2[], int size2, int intersection[], int intersection_bufsize, int *size3);

int main() {
    int arr1[] = {1, 2, 3, 4, 5};
    int size1 = sizeof(arr1) / sizeof(arr1[0]);

    int arr2[] = {3, 4, 5, 6, 7};
    int size2 = sizeof(arr2) / sizeof(arr2[0]);

    int intersection[10];
    int size3 = 0; // Initialize to avoid uninitialized use

    // Pass the buffer size to the function
    findIntersection(arr1, size1, arr2, size2, intersection, sizeof(intersection)/sizeof(intersection[0]), &size3);

    printf("Intersection: ");
    for (int i = 0; i < size3; i++) {
        printf("%d ", intersection[i]);
    }
    printf("\n");

    return 0;
}

void findIntersection(int arr1[], int size1, int arr2[], int size2, int intersection[], int intersection_bufsize, int *size3) {
    int k = 0;
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            if (arr1[i] == arr2[j]) {
                // Check for duplicates in intersection
                int duplicate = 0;
                for (int m = 0; m < k; m++) {
                    if (intersection[m] == arr1[i]) {
                        duplicate = 1;
                        break;
                    }
                }
                if (!duplicate) {
                    if (k < intersection_bufsize) {
                        intersection[k++] = arr1[i];
                    } else {
                        // Buffer full, stop adding more elements
                        // Optionally, print a warning or handle error
                        goto done;
                    }
                }
            }
        }
    }
done:
    *size3 = k; // Always set size3
}