#include <stdio.h>

void findIntersection(int arr1[], int size1, int arr2[], int size2, int intersection[], int *size3);

int main() {
    int arr1[] = {1, 2, 3, 4, 5};
    int size1 = sizeof(arr1) / sizeof(arr1[0]);

    int arr2[] = {3, 4, 5, 6, 7};
    int size2 = sizeof(arr2) / sizeof(arr2[0]);

    // Allocate intersection with the smallest size of the two arrays
    int intersection[size1 < size2 ? size1 : size2];
    int size3 = 0;  // Initialize to zero

    findIntersection(arr1, size1, arr2, size2, intersection, &size3);

    printf("Intersection: ");
    for (int i = 0; i < size3; i++) {
        printf("%d ", intersection[i]);
    }

    return 0;
}

void findIntersection(int arr1[], int size1, int arr2[], int size2, int intersection[], int *size3) {
    int k = 0;
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            if (arr1[i] == arr2[j]) {
                intersection[k] = arr1[i];
                k++;
                break;  // Break once you find the intersection to avoid duplicates
            }
        }
    }
    *size3 = k;
}