#include "malutilities.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <regex>
#include <stdexcept>

// Class to represent a Factorio recipe
class Recipe {
public:
    Recipe(const std::string& member) : rawMember(member) {}

    std::string getName() const;
    std::string getIngredients() const;
    std::string toString() const;

private:
    std::string extractNestedBraces(const std::string& input) const;
    std::string parseIngredients(const std::string& ingredientsData) const;
    std::string removeCraftingMachineTint(const std::string& inputMember) const;

    std::string rawMember;
};

// Function to remove leading whitespace
void removeLeadingWhitespace(std::string& text) {
    // Implementation remains the same
}

// Function to remove line starting with a specific sequence
void removeLineStartingWith(std::string& text, const std::string& sequence) {
    // Implementation remains the same
}

// Function to remove quotes from a string
std::string removeQuotes(const std::string& input) {
    // Implementation remains the same
}

// Function to ensure newline after brace
void ensureNewlineAfterBrace(std::string& text) {
    // Implementation remains the same
}

// Function to insert newline between consecutive brackets
void insertNewlineBetweenBrackets(std::string& input) {
    // Implementation remains the same
}

// Function to remove newline after equals sign
void removeNewlineAfterEquals(std::string& text) {
    // Implementation remains the same
}

// Function to generate output from individual members
std::string generateOutput(const std::vector<Recipe>& recipes) {
    std::ostringstream rawStringStream;
    rawStringStream << "{\n";
    for (size_t i = 0; i < recipes.size(); ++i) {
        rawStringStream << removeQuotes(recipes[i].getName()) << '\n'
                        << removeQuotes(recipes[i].getIngredients()) << "\n\n";
    }
    rawStringStream << "}\n";
    return rawStringStream.str();
}

// Function to extract individual members from raw string
std::vector<std::string> extractIndividualMembers(const std::string& input) {
    // Implementation remains the same
}

// Secure file reading function with error handling and validation
std::string readFileIntoRawString(const std::string& filename) {
    // Validate filename: only allow alphanumeric, dot, dash, and underscore
    if (!std::regex_match(filename, std::regex("^[\\w\\-.]+$"))) {
        throw std::invalid_argument("Invalid filename: " + filename);
    }
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    if (file.fail() && !file.eof()) {
        throw std::runtime_error("Error reading file: " + filename);
    }
    return ss.str();
}

// Secure file writing function with error handling
void writeToTxtFile(const std::string& data, const std::string& filename) {
    // Validate filename: only allow alphanumeric, dot, dash, and underscore
    if (!std::regex_match(filename, std::regex("^[\\w\\-.]+$"))) {
        throw std::invalid_argument("Invalid filename: " + filename);
    }
    std::ofstream file(filename, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    file << data;
    if (!file) {
        throw std::runtime_error("Error writing to file: " + filename);
    }
}

// Main function
int main() {
    std::string filename = "TextFile1.txt";
    std::string rawString;

    try {
        rawString = readFileIntoRawString(filename);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    // Cleaning up raw string
    removeLeadingWhitespace(rawString);
    ensureNewlineAfterBrace(rawString);
    removeLeadingWhitespace(rawString);

    std::vector<std::string> catsToRemove = { "type", "category", "enabled", "order", "allow_decomposition", "main_product", "subgroup", "requester_paste_multiplier", "icon" };
    for (const auto& category : catsToRemove) {
        removeLineStartingWith(rawString, category);
    }
    removeNewlineAfterEquals(rawString);

    // Extract individual members (recipes)
    std::vector<std::string> individualMembers = extractIndividualMembers(rawString);

    // Create Recipe objects
    std::vector<Recipe> recipes;
    for (const auto& member : individualMembers) {
        recipes.emplace_back(member);
    }

    // Generate output
    std::string output = generateOutput(recipes);

    // Output to console (sanitize output if necessary)
    std::cout << output;

    // Output to file with error handling
    try {
        writeToTxtFile(output, "roughCut.txt");
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

// Class member function implementations

std::string Recipe::extractNestedBraces(const std::string& input) const {
    // Implementation remains the same
}

std::string Recipe::parseIngredients(const std::string& ingredientsData) const {
    // Implementation remains the same
}

std::string Recipe::removeCraftingMachineTint(const std::string& inputMember) const {
    // Implementation remains the same
}

std::string Recipe::getName() const {
    // Implementation remains the same
}

std::string Recipe::getIngredients() const {
    // Implementation remains the same
}

std::string Recipe::toString() const {
    // Implementation remains the same
}