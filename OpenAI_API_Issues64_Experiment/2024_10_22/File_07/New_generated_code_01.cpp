#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include "student.h"
#include "bubble.h"

int main()
{
    // Open the grades file for reading
    std::ifstream infile("grades");
    if (!infile) {
        std::cerr << "Could not open the file." << std::endl;
        return 1;
    }

    try {
        // Skip the first line (CSCE1040)
        std::string line;
        std::getline(infile, line);

        // Vector of unique pointers managing student memory
        std::vector<std::unique_ptr<student>> student_ptrs;

        // Read and initialize each student
        std::string first, last;
        int exam1, exam2, exam3;
        while (infile >> first >> last >> exam1 >> exam2 >> exam3) {
            auto s = std::make_unique<student>();
            s->first = first;
            s->last = last;
            s->exam1 = exam1;
            s->exam2 = exam2;
            s->exam3 = exam3;
            s->mean = (exam1 + exam2 + exam3) / 3.0;
            student_ptrs.push_back(std::move(s));
        }

        // Close the file (this happens automatically at the end of scope, but is explicit here for clarity)
        infile.close();

        // Sort the students by mean score
        bubble(student_ptrs.data(), student_ptrs.size());

        // Display the sorted students
        for (const auto& s : student_ptrs) {
            std::cout << s->first << " " << s->last << " " << s->mean << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}