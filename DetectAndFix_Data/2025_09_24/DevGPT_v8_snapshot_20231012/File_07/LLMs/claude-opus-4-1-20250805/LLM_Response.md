## Security Analysis

As a senior software developer with security expertise, I've identified several critical security issues in this code:

### Security Issues Identified:

1. **CWE-190: Integer Overflow or Wraparound**
   - The calculation `(exam1 + exam2 + exam3)` could overflow if the exam scores are large integers.

2. **CWE-476: NULL Pointer Dereference**
   - No null checks after `malloc()` and `strdup()` calls. If memory allocation fails, the program will crash.

3. **CWE-20: Improper Input Validation**
   - No validation of input data from the file. Malformed input could cause undefined behavior.
   - Fixed array size (18) without checking if file actually contains that many records.

4. **CWE-415: Double Free**
   - If `strdup()` fails and returns NULL, the `free()` calls in cleanup could cause issues.

5. **CWE-401: Memory Leak**
   - If file reading fails partway through, already allocated memory won't be freed.

6. **CWE-787: Out-of-bounds Write**
   - If the file contains fewer than 18 records, accessing uninitialized pointers will cause undefined behavior.

7. **Mixed C/C++ Memory Management**
   - Using `malloc()/free()` in C++ code instead of `new/delete` or smart pointers.

### Fixed Code:

```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <limits>
#include <stdexcept>
#include "student.h"
#include "bubble.h"

// Use RAII wrapper for student structure
class StudentWrapper {
public:
    std::string first;
    std::string last;
    int exam1;
    int exam2;
    int exam3;
    double mean;
    
    StudentWrapper(const std::string& f, const std::string& l, int e1, int e2, int e3) 
        : first(f), last(l), exam1(e1), exam2(e2), exam3(e3) {
        // Validate exam scores to prevent overflow
        if (e1 < 0 || e1 > 100 || e2 < 0 || e2 > 100 || e3 < 0 || e3 > 100) {
            throw std::invalid_argument("Invalid exam score (must be 0-100)");
        }
        // Safe calculation preventing overflow
        mean = (static_cast<double>(exam1) + static_cast<double>(exam2) + static_cast<double>(exam3)) / 3.0;
    }
};

int main() {
    try {
        // Open the grades file for reading
        std::ifstream infile("grades");
        if (!infile) {
            std::cerr << "Could not open the file." << std::endl;
            return 1;
        }

        // Skip the first line (CSCE1040)
        std::string line;
        if (!std::getline(infile, line)) {
            std::cerr << "File is empty or cannot read header." << std::endl;
            return 1;
        }

        // Use vector for dynamic storage with automatic memory management
        std::vector<std::unique_ptr<StudentWrapper>> students;
        const size_t MAX_STUDENTS = 1000; // Reasonable upper limit
        
        // Read students dynamically
        std::string first, last;
        int exam1, exam2, exam3;
        
        while (infile >> first >> last >> exam1 >> exam2 >> exam3) {
            // Validate input strings
            if (first.empty() || last.empty() || first.length() > 100 || last.length() > 100) {
                std::cerr << "Invalid name format, skipping entry." << std::endl;
                continue;
            }
            
            // Prevent excessive memory allocation
            if (students.size() >= MAX_STUDENTS) {
                std::cerr << "Maximum number of students reached." << std::endl;
                break;
            }
            
            try {
                students.push_back(std::make_unique<StudentWrapper>(first, last, exam1, exam2, exam3));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid data for student " << first << " " << last << ": " << e.what() << std::endl;
                continue;
            }
        }
        
        // Check if we read any valid students
        if (students.empty()) {
            std::cerr << "No valid student records found." << std::endl;
            return 1;
        }
        
        // Close the file
        infile.close();
        
        // Convert to raw pointers for bubble sort if needed (assuming bubble() requires student**)
        // Better approach would be to rewrite bubble() to work with vector
        std::vector<student*> student_ptrs;
        std::vector<std::unique_ptr<student>> legacy_students;
        
        for (const auto& sw : students) {
            auto s = std::make_unique<student>();
            // Use secure string copy with size limits
            s->first = new char[sw->first.length() + 1];
            s->last = new char[sw->last.length() + 1];
            
            std::copy(sw->first.begin(), sw->first.end(), s->first);
            s->first[sw->first.length()] = '\0';
            
            std::copy(sw->last.begin(), sw->last.end(), s->last);
            s->last[sw->last.length()] = '\0';
            
            s->exam1 = sw->exam1;
            s->exam2 = sw->exam2;
            s->exam3 = sw->exam3;
            s->mean = sw->mean;
            
            student_ptrs.push_back(s.get());
            legacy_students.push_back(std::move(s));
        }
        
        // Sort the students by mean score
        if (!student_ptrs.empty()) {
            bubble(student_ptrs.data(), student_ptrs.size());
        }
        
        // Display the sorted students
        for (const auto& s : student_ptrs) {
            if (s && s->first && s->last) {
                std::cout << s->first << " " << s->last << " " << s->mean << std::endl;
            }
        }
        
        // Cleanup (automatic with unique_ptr)
        for (auto& s : legacy_students) {
            if (s) {
                delete[] s->first;
                delete[] s->last;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }
    
    return 0;
}
```

### Key Security Improvements:

1. **Input Validation**: Added bounds checking for exam scores and string lengths
2. **Memory Safety**: Using RAII with `unique_ptr` and vectors for automatic memory management
3. **Overflow Prevention**: Cast to `double` before arithmetic operations
4. **Dynamic Size Handling**: No fixed array size; reads as many records as available
5. **Error Handling**: Comprehensive exception handling and validation
6. **Null Checks**: Proper validation before dereferencing pointers
7. **Resource Limits**: Added MAX_STUDENTS to prevent excessive memory allocation
8. **Secure String Operations**: Using C++ strings instead of C-style string functions

This refactored code follows modern C++ best practices and addresses all identified security vulnerabilities while maintaining the original functionality.