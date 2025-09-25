#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct Student {
    std::string first;
    std::string last;
    int exam1{};
    int exam2{};
    int exam3{};
    double mean{};
};

static bool parse_line(const std::string& line, Student& out) {
    // Expect: first last exam1 exam2 exam3
    std::istringstream iss(line);
    std::string first, last;
    long long e1, e2, e3;

    if (!(iss >> first >> last >> e1 >> e2 >> e3)) {
        return false; // CWE-20 mitigation: reject malformed rows
    }

    // Validate numeric ranges to mitigate overflow/garbage inputs (adjust limits as needed)
    auto in_range = [](long long v) { return v >= 0 && v <= 100; };
    if (!in_range(e1) || !in_range(e2) || !in_range(e3)) {
        return false;
    }

    out.first = std::move(first);
    out.last  = std::move(last);
    out.exam1 = static_cast<int>(e1);
    out.exam2 = static_cast<int>(e2);
    out.exam3 = static_cast<int>(e3);

    // CWE-190 mitigation: use double and promote before sum
    out.mean = (static_cast<double>(out.exam1) +
                static_cast<double>(out.exam2) +
                static_cast<double>(out.exam3)) / 3.0;
    return true;
}

int main() {
    using namespace std;
    namespace fs = std::filesystem;

    const std::string path = "grades";

    // CWE-73 mitigation: basic validation of input path
    try {
        if (!fs::exists(path) || !fs::is_regular_file(fs::status(path))) {
            cerr << "Input file not found or not a regular file.\n";
            return 1;
        }
    } catch (const fs::filesystem_error& e) {
        cerr << "Filesystem error: " << e.what() << "\n";
        return 1;
    }

    ifstream infile(path);
    if (!infile) {
        cerr << "Could not open the file.\n";
        return 1;
    }

    // Skip header line (e.g., CSCE1040)
    std::string header;
    if (!std::getline(infile, header)) {
        cerr << "Failed to read header line.\n";
        return 1;
    }

    vector<Student> students;
    students.reserve(18); // optional, if you expect ~18 rows

    string line;
    size_t line_no = 1; // header consumed
    while (std::getline(infile, line)) {
        ++line_no;
        if (line.empty()) {
            continue;
        }
        Student s;
        if (!parse_line(line, s)) {
            cerr << "Skipping malformed data on line " << line_no << ".\n";
            continue; // CWE-252/CWE-20 mitigation: only accept valid rows
        }
        students.push_back(std::move(s));
    }

    if (students.empty()) {
        cerr << "No valid student records found.\n";
        return 1;
    }

    // Sort by mean descending; avoids custom bubble sort
    std::sort(students.begin(), students.end(),
              [](const Student& a, const Student& b) { return a.mean > b.mean; });

    for (const auto& s : students) {
        cout << s.first << ' ' << s.last << ' ' << s.mean << '\n';
    }

    return 0;
}