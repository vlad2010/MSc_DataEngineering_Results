#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm> // For std::all_of

// Removed unused global variable deadEnds

bool wordLadder(const std::string& word, const std::string& wordFinal,
                const std::vector<std::string>& words,
                std::vector<std::string>& ladder);

bool isRedundant(const std::string& word, const std::vector<std::string>& ladder);

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

    inFile.open(argv[1]);
    if (!inFile.is_open()) { // CWE-252: Check file open
        std::cerr << "Error: Could not open file " << argv[1] << "\n";
        return 1;
    }

    // Read word bank from file (CWE-665: Proper file reading)
    std::string input;
    while (inFile >> input) {
        wordBank.push_back(input);
    }
    inFile.close();

    // Take input of starting and final words
    std::cout << "Enter start word: ";
    std::cin >> word;
    std::cout << "Enter target word: ";
    std::cin >> target;

    // CWE-20: Validate input words exist in word bank
    auto is_in_bank = [&](const std::string& w) {
        return std::find(wordBank.begin(), wordBank.end(), w) != wordBank.end();
    };
    if (!is_in_bank(word) || !is_in_bank(target)) {
        std::cerr << "Error: Start or target word not in word bank.\n";
        return 1;
    }

    // CWE-119: Ensure all words are the same length
    size_t word_len = word.length();
    if (target.length() != word_len) {
        std::cerr << "Error: Start and target words must be the same length.\n";
        return 1;
    }
    if (!std::all_of(wordBank.begin(), wordBank.end(), [word_len](const std::string& w){ return w.length() == word_len; })) {
        std::cerr << "Error: All words in the word bank must be the same length.\n";
        return 1;
    }

    ladder.push_back(word);
    bool found = false;
    size_t past_size = 0; // Use size_t for unsigned comparison

    // Find word ladder
    while (!found) {
        found = wordLadder(word, target, wordBank, ladder);

        if (found) {
            break;
        } else {
            if (past_size == ladder.size()) {
                std::cout << "Solution Does Not Exist\n";
                break;
            }
            if (!ladder.empty()) ladder.pop_back();
            past_size = ladder.size();
        }
    }

    // Clear ladder if the final word is not reached
    if (!ladder.empty() && ladder.back() != target)
        ladder.clear();

    // Print word ladder
    if (!ladder.empty())
        for (const auto& w : ladder)
            std::cout << w << '\n';

    return 0;
}

bool wordLadder(const std::string& word, const std::string& wordFinal,
                const std::vector<std::string>& words,
                std::vector<std::string>& ladder)
{
    // Base case: final word reached
    if (word == wordFinal) {
        return true;
    }

    // General case: find adjacent words in the word bank
    for (const auto& strcomp : words) {
        if (isRedundant(strcomp, ladder)) {
            continue;
        }

        // CWE-119: Ensure safe access
        if (strcomp.length() != word.length()) continue;

        int counter = 0;
        for (size_t i = 0; i < word.length(); i++) {
            if (word[i] == strcomp[i]) 
                counter++;
        }

        if (counter == static_cast<int>(word.length()) - 1) {
            ladder.push_back(strcomp);
            bool found = wordLadder(strcomp, wordFinal, words, ladder);
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
    // Check if the word is already in the ladder
    return std::find(ladder.begin(), ladder.end(), word) != ladder.end();
}