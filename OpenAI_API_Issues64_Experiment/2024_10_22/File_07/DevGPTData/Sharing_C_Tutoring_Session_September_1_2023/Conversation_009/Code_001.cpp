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
    // For simplicity, let's assume we have 3 students.
    student students[3];
    students[0] = {"Erica", "Sanders", 75, 89, 67, 77.0};
    students[1] = {"Kelley", "Cummings", 74, 70, 79, 74.3};
    students[2] = {"Jamie", "Reynolds", 64, 52, 66, 60.7};

    // Create an array of pointers to students
    student* student_ptrs[3];
    for (int i = 0; i < 3; ++i) {
        student_ptrs[i] = &students[i];
    }

    // Sort the students by mean score
    bubble(student_ptrs, 3);

    // Display the sorted students
    for (int i = 0; i < 3; ++i) {
        std::cout << student_ptrs[i]->first << " " << student_ptrs[i]->last << " " << student_ptrs[i]->mean << std::endl;
    }

    return 0;
}
