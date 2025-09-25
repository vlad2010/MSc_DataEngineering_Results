#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_RECURSION_DEPTH 1000
#define BUFFER_SIZE 2048

static int recursion_depth = 0;

const char* days[] = {
    "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"
};

const char* gifts[] = {
    "And a partridge in a pear tree.",
    "Two turtle doves,",
    "Three French hens,",
    "Four calling birds,",
    "Five golden rings,",
    "Six geese a-laying,",
    "Seven swans a-swimming,",
    "Eight maids a-milking,",
    "Nine ladies dancing,",
    "Ten lords a-leaping,",
    "Eleven pipers piping,",
    "Twelve drummers drumming,"
};

void print_verse(int day) {
    // Input validation
    if (day < 0 || day >= 12) {
        fprintf(stderr, "Invalid day number: %d\n", day);
        return;
    }
    
    // Print verse header
    printf("On the %s day of Christmas, my true love sent to me:\n", days[day]);
    
    // Print gifts in reverse order
    for (int i = day; i >= 0; i--) {
        if (i == 0 && day == 0) {
            printf("A partridge in a pear tree.\n");
        } else {
            printf("%s\n", gifts[i]);
        }
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    // Proper argument validation
    int start_day = 0;
    int end_day = 11;
    
    // Optional: parse command line arguments safely
    if (argc > 1) {
        char *endptr;
        long val = strtol(argv[1], &endptr, 10);
        
        // Check for conversion errors
        if (endptr == argv[1] || *endptr != '\0') {
            fprintf(stderr, "Invalid argument: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
        
        // Range validation
        if (val < 1 || val > 12) {
            fprintf(stderr, "Day must be between 1 and 12\n");
            return EXIT_FAILURE;
        }
        
        start_day = (int)(val - 1);
        end_day = start_day;
    }
    
    // Print the song
    for (int day = start_day; day <= end_day; day++) {
        print_verse(day);
    }
    
    return EXIT_SUCCESS;
}