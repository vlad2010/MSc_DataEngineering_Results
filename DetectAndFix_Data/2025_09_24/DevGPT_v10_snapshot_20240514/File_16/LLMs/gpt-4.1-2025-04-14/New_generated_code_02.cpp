#include <iostream>
#include <fstream>
#include <string>
#include <regex>

int main() {
    std::string filename;
    int cases;

    std::cout << "Enter filename: ";
    std::getline(std::cin, filename);

    // Validate filename: allow only alphanumeric, underscore, dash, and dot, no path separators
    std::regex valid_filename("^[\\w\\-.]+$");
    if (!std::regex_match(filename, valid_filename)) {
        std::cerr << "Invalid filename. Only alphanumeric characters, underscores, dashes, and dots are allowed." << std::endl;
        return 1;
    }

    // Prevent path traversal by rejecting filenames with ".."
    if (filename.find("..") != std::string::npos) {
        std::cerr << "Invalid filename: path traversal detected." << std::endl;
        return 1;
    }

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "'." << std::endl;
        return 1;
    }

    if (!(infile >> cases)) {
        std::cerr << "Error: Failed to read 'cases' from file." << std::endl;
        return 1;
    }

    std::cout << "Cases read: " << cases << std::endl;
    // ... rest of your code ...
    return 0;
}