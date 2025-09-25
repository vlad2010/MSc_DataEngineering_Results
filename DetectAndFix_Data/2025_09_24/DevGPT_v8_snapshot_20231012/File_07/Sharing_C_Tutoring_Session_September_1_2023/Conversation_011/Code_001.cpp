#include <iostream>
#include <stdlib.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include "student.h"
#include "bubble.h"

int main()
{
    // Dynamic memory allocation for 18 student pointers
    student **student_ptrs = (student **)malloc(18 * sizeof(student *));

    // Dynamic memory allocation and initialization for each student
    for (int i = 0; i < 18; ++i) {
        student_ptrs[i] = (student *)malloc(sizeof(student));
    }

    // Initialize students. Ideally, this would be done by reading from a file or database.
    // For demonstration purposes, the initialization is hardcoded.
    student_ptrs[0]->first = "Erica";      student_ptrs[0]->last = "Sanders";    student_ptrs[0]->exam1 = 75; student_ptrs[0]->exam2 = 89; student_ptrs[0]->exam3 = 67; student_ptrs[0]->mean = 77.0;
    student_ptrs[1]->first = "Kelley";     student_ptrs[1]->last = "Cummings";   student_ptrs[1]->exam1 = 74; student_ptrs[1]->exam2 = 70; student_ptrs[1]->exam3 = 79; student_ptrs[1]->mean = 74.3;
    student_ptrs[2]->first = "Jamie";      student_ptrs[2]->last = "Reynolds";   student_ptrs[2]->exam1 = 64; student_ptrs[2]->exam2 = 52; student_ptrs[2]->exam3 = 66; student_ptrs[2]->mean = 60.7;
    // ... continue initializing all 18 students

    // Sort the students by mean score
    bubble(student_ptrs, 18);

    // Display the sorted students
    for (int i = 0; i < 18; ++i) {
        std::cout << student_ptrs[i]->first << " " << student_ptrs[i]->last << " " << student_ptrs[i]->mean << std::endl;
    }

    // Free the dynamically allocated memory
    for (int i = 0; i < 18; ++i) {
        free(student_ptrs[i]);
    }
    free(student_ptrs);

    return 0;
}
