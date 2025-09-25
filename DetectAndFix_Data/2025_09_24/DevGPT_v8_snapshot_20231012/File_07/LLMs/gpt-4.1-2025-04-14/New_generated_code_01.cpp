#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

// Example student struct
struct student {
    std::string first;
    std::string last;
    int exam1;
    int exam2;
    int exam3;
    double mean;
};

// Bubble sort function for students by mean (replace with std::sort in real code)
void bubble(std::vector<student>& students) {
    size_t n = students.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n - i - 1; ++j) {
            if (students[j].mean < students[j + 1].mean) {
                std::swap(students[j], students[j + 1]);
            }
        }
    }
}

int main()
{
    // Open the grades file for reading
    std::ifstream infile("grades");
    if (!infile) {
        std::cerr << "Could not open the file." << std::endl;
        return 1;
    }

    // Skip the first line (CSCE1040)
    std::string line;
    if (!std::getline(infile, line)) {
        std::cerr << "File is empty or malformed." << std::endl;
        return 1;
    }

    std::vector<student> students;
    std::string first, last;
    int exam1, exam2, exam3;
    int line_num = 2; // Start from line 2 (after header)
    while (infile >> first >> last >> exam1 >> exam2 >> exam3) {
        // Input validation
        if (exam1 < 0 || exam2 < 0 || exam3 < 0) {
            std::cerr << "Invalid exam score at line " << line_num << std::endl;
            continue;
        }
        student s;
        s.first = first;
        s.last = last;
        s.exam1 = exam1;
        s.exam2 = exam2;
        s.exam3 = exam3;
        s.mean = (exam1 + exam2 + exam3) / 3.0;
        students.push_back(s);
        ++line_num;
    }

    if (students.empty()) {
        std::cerr << "No valid student records found." << std::endl;
        return 1;
    }

    // Sort the students by mean score (descending)
    bubble(students);

    // Display the sorted students
    for (const auto& s : students) {
        std::cout << s.first << " " << s.last << " " << s.mean << std::endl;
    }

    return 0;
}