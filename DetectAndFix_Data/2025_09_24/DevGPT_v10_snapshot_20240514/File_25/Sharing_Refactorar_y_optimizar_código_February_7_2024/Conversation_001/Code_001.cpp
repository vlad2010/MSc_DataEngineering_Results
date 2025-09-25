#include "malutilities.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <regex>

// Function prototypes
std::string extractNestedBraces(const std::string& input);
std::vector<std::string> extractIndividualMembers(const std::string& input);
std::string getNormal(const std::string& inputMember);
std::string getIngredients(const std::string& inputMember);
std::string getRecipeName(const std::string& inputMember);
std::string parseIngredients(const std::string& ingredientsData);
void removeLeadingWhitespace(std::string& text);
void removeLineStartingWith(std::string& text, const std::string& sequence);
std::string removeQuotes(const std::string& input);
void ensureNewlineAfterBrace(std::string& text);
void insertNewlineBetweenBrackets(std::string& input);
void removeNewlineAfterEquals(std::string& text);
std::string removeCraftingMachineTint(const std::string& inputMember);
std::string generateOutput(const std::vector<std::string>& individualMembers);

int main() {
    std::string filename = "TextFile1.txt";
    std::string rawString = readFileIntoRawString(filename);

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

    // Generate output
    std::string output = generateOutput(individualMembers);

    // Output to console
    std::cout << output;

    // Output to file
    writeToTxtFile(output, "roughCut.txt");

    return 0;
}

// Function to extract nested braces
std::string extractNestedBraces(const std::string& input) {
    // Implementation remains the same
}

// Function to extract individual members from raw string
std::vector<std::string> extractIndividualMembers(const std::string& input) {
    // Implementation remains the same
}

// Function to get normal recipe
std::string getNormal(const std::string& inputMember) {
    // Implementation remains the same
}

// Function to get ingredients
std::string getIngredients(const std::string& inputMember) {
    // Implementation remains the same
}

// Function to get recipe name
std::string getRecipeName(const std::string& inputMember) {
    // Implementation remains the same
}

// Function to parse ingredients data
std::string parseIngredients(const std::string& ingredientsData) {
    // Implementation remains the same
}

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

// Function to remove crafting machine tint
std::string removeCraftingMachineTint(const std::string& inputMember) {
    // Implementation remains the same
}

// Function to generate output from individual members
std::string generateOutput(const std::vector<std::string>& individualMembers) {
    std::ostringstream rawStringStream;
    rawStringStream << "{\n";
    for (size_t i = 0; i < individualMembers.size(); ++i) {
        std::string nameOutput = getRecipeName(individualMembers[i]);
        std::string ingredientsOutput = getIngredients(individualMembers[i]);

        rawStringStream << removeQuotes(nameOutput) << '\n' << removeQuotes(ingredientsOutput) << "\n\n";
    }
    rawStringStream << "}\n";

    return rawStringStream.str();
}