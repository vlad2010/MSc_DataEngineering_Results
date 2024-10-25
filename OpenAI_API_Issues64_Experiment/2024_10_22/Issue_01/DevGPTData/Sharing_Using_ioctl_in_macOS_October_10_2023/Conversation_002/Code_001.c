#include <stdio.h>

int main() {
    char buffer[256];
    while (1) {
        printf("Enter something: ");
        fgets(buffer, sizeof(buffer), stdin);
        printf("You entered: %s", buffer);
    }
    return 0;
}
