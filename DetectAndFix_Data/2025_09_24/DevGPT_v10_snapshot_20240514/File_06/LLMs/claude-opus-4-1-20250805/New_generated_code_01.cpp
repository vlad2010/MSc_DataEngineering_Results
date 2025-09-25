/**
 * @file word_ladder.cpp
 * @brief Implements a function to find a word ladder between two words.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>

// Constants for security limits
const size_t MAX_WORD_LENGTH = 100;
const size_t MAX_WORDS_IN_BANK = 100000;
const size_t MAX_RECURSION_DEPTH = 1000;
const size_t MAX_ITERATIONS = 10000;

/// Global variable to store dead ends encountered during ladder traversal.
std::vector<std::string> deadEnds;

bool wordLadder(std::string word, std::string wordFinal,
                const std::vector<std::string>& words,
                std::vector<std::string>& ladder,
                size_t depth = 0);

bool isRedundant(const std::string& word, const std::vector<std::string>& ladder);
void deleteDeadEnds(std::vector<std::string>& ladder);
void reverseLadder(std::vector<std::string>& ladder);
bool isValidWord(const std::string& word);

int main(int argc, char* argv[])
{
    std::vector<std::string> wordBank;
    std::vector<std::string> ladder;
    std::ifstream inFile;
    std::string word, target;
    int status = 0;

    // Check command-line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <wordbank_file>\n";
        return 1;
    }

    // Open and validate file
    inFile.open(argv[1]);
    if (!inFile.is_open()) {
        std::cerr << "Error: Cannot open file '" << argv[1] << "'\n";
        return 1;
    }

    // Read word bank from file with security limits
    std::string input;
    while (inFile >> input && wordBank.size() < MAX_WORDS_IN_BANK) {
        if (isValidWord(input)) {
            wordBank.push_back(input);
        } else {
            std::cerr << "Warning: Skipping invalid word: " << input << "\n";
        }
    }
    inFile.close();

    if (wordBank.empty()) {
        std::cerr << "Error: Word bank is empty\n";
        return 1;
    }

    // Take input of starting and final words with validation
    if (!(std::cin >> word >> target)) {
        std::cerr << "Error: Failed to read input words\n";
        return 1;
    }

    if (!isValidWord(word) || !isValidWord(target)) {
        std::cerr << "Error: Invalid input words\n";
        return 1;
    }

    if (word.length() != target.length()) {
        std::cerr << "Error: Starting and target words must have the same length\n";
        return 1;
    }

    ladder.push_back(word);
    bool found = false;
    size_t past_size = 0;
    size_t iterations = 0;

    // Find word ladder with iteration limit
    while (!found && iterations < MAX_ITERATIONS) {
        found = wordLadder(word, target, wordBank, ladder);

        if (found) {
            break;
        } else {
            if (past_size == ladder.size()) {
                std::cout << "Solution Does Not Exist\n";
                break;
            }
            if (!ladder.empty()) {
                ladder.pop_back();
            }
            past_size = ladder.size();
        }
        iterations++;
    }

    if (iterations >= MAX_ITERATIONS) {
        std::cerr << "Error: Maximum iterations reached\n";
        return 1;
    }

    // Verify and print word ladder
    if (!ladder.empty() && ladder.back() == target) {
        for (const auto& w : ladder) {
            std::cout << w << '\n';
        }
    }

    return 0;
}

bool wordLadder(std::string word, std::string wordFinal,
                const std::vector<std::string>& words,
                std::vector<std::string>& ladder,
                size_t depth)
{
    // Prevent stack overflow
    if (depth > MAX_RECURSION_DEPTH) {
        return false;
    }

    // Base case: final word reached
    if (word == wordFinal) {
        return true;
    }

    // General case: find adjacent words in the word bank
    for (const auto& strcomp : words) {
        // Skip if word is already in ladder
        if (isRedundant(strcomp, ladder)) {
            continue;
        }

        // Check if words have same length before comparing
        if (word.length() != strcomp.length()) {
            continue;
        }

        // Count matching characters
        size_t counter = 0;
        for (size_t i = 0; i < word.length(); i++) {
            if (word[i] == strcomp[i]) {
                counter++;
            }
        }

        // If exactly one character differs, try this path
        if (counter == word.length() - 1) {
            ladder.push_back(strcomp);
            bool found = wordLadder(strcomp, wordFinal, words, ladder, depth + 1);
            if (found) {
                return true;
            } else {
                ladder.pop_back();
            }
        }
    }

    return false;
}

bool isRedundant(const std::string& word, const std::vector<std::string>& ladder)
{
    return std::find(ladder.begin(), ladder.end(), word) != ladder.end();
}

bool isValidWord(const std::string& word)
{
    if (word.empty() || word.length() > MAX_WORD_LENGTH) {
        return false;
    }
    
    // Check if all characters are alphabetic
    return std::all_of(word.begin(), word.end(), 
                       [](char c) { return std::isalpha(c); });
}

void deleteDeadEnds(std::vector<std::string>& ladder)
{
    // Functionality not implemented
}

void reverseLadder(std::vector<std::string>& ladder)
{
    std::reverse(ladder.begin(), ladder.end());
}