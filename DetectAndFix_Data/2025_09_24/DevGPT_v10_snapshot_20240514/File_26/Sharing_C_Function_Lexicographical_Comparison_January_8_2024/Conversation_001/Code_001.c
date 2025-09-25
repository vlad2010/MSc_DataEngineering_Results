#include <stdio.h>
#include <stdbool.h>

struct Record {
    int id;
    char name[15];
    char surname[25];
    char address[50];
};

bool shouldSwap(struct Record* rec1, struct Record* rec2) {
    int nameComparison = strcmp(rec1->name, rec2->name);

    if (nameComparison < 0) {
        // rec1's name comes before rec2's name
        return true;
    } else if (nameComparison == 0) {
        // Names are equal, compare surnames
        int surnameComparison = strcmp(rec1->surname, rec2->surname);

        if (surnameComparison < 0) {
            // rec1's surname comes before rec2's surname
            return true;
        }
    }

    // No need to swap
    return false;
}

int main() {
    // Example usage
    struct Record record1 = {1, "John", "Doe", "123 Main St"};
    struct Record record2 = {2, "Alice", "Smith", "456 Oak St"};

    if (shouldSwap(&record1, &record2)) {
        printf("Swapping records.\n");
        // Perform swap operation here if needed
    } else {
        printf("No need to swap records.\n");
    }

    // Hello world message
    printf("Hello world.\n");

    return 0;
}