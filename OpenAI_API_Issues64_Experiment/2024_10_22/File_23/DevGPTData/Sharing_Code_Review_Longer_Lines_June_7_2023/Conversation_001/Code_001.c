#include <stdio.h>

#define MIN_LINE_LENGTH 80
#define MAX_LINE_LENGTH 1000

int get_line(char line[], int max) {
  int c, i;

  for (i = 0; i < max - 1 && (c = getchar()) != EOF && c != '\n'; i++) {
    line[i] = c;
  }
  
  if (c == '\n') {
    line[i] = c;
    ++i;
  }
  
  line[i] = '\0';  // Ensure the line is null-terminated.
  
  return i;
}

int main() {
  printf("=== This program will print all the lines that exceed 80 characters, "
         "exit by CTRL+D ===\n");

  char line[MAX_LINE_LENGTH];
  int line_length;

  while ((line_length = get_line(line, MAX_LINE_LENGTH)) > 0) {
    if (line_length > MIN_LINE_LENGTH) {
      printf("%s", line);
    }
  }

  return 0;
}
