#include <stdio.h>

void findIntersection(int arr1[], int size1, int arr2[], int size2, int intersection[], int *size3);

int main() {
    int arr1[] = {1, 2, 3, 4, 5};
    int size1 = sizeof(arr1) / sizeof(arr1[0]);

    int arr2[] = {3, 4, 5, 6, 7};
    int size2 = sizeof(arr2) / sizeof(arr2[0]);

    int intersection[10];
    int size3;

    findIntersection(arr1, size1, arr2, size2, intersection, &size3);

    printf("Intersection: ");
    for (int i = 0; i < size3; i++) {
        printf("%d ", intersection[i]);
    }

    return 0;
}

void findIntersection(int arr1[], int size1, int arr2[], int size2, int intersection[], int *size3) {
    // Your code to find the intersection goes here
}
