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

    // Read word bank from file (CWE-665: Proper input reading)
    std::string input;
    while (inFile >> input) {
        wordBank.push_back(input);
    }
    inFile.close();

    if (wordBank.empty()) {
        std::cerr << "Error: Word bank is empty.\n";
        return 1;
    }

    // Take input of starting and final words
    std::cout << "Enter start word: ";
    if (!(std::cin >> word)) {
        std::cerr << "Error: Failed to read start word.\n";
        return 1;
    }
    std::cout << "Enter target word: ";
    if (!(std::cin >> target)) {
        std::cerr << "Error: Failed to read target word.\n";
        return 1;
    }

    // CWE-120: Check input lengths
    size_t wordLength = wordBank[0].length();
    if (word.length() != wordLength || target.length() != wordLength) {
        std::cerr << "Error: Start and target words must be length " << wordLength << ".\n";
        return 1;
    }
    // Ensure all words in wordBank are the same length
    if (!std::all_of(wordBank.begin(), wordBank.end(),
        [wordLength](const std::string& w){ return w.length() == wordLength; })) {
        std::cerr << "Error: All words in word bank must be of length " << wordLength << ".\n";
        return 1;
    }

    ladder.push_back(word);
    bool found = false;
    int past_size = -1;
    int max_attempts = 10000; // Prevent infinite loop (CWE-835)
    int attempts = 0;

    // Find word ladder
    while (!found && attempts < max_attempts) {
        found = wordLadder(word, target, wordBank, ladder);

        if (found) {
            break;
        } else {
            if (past_size == static_cast<int>(ladder.size())) {
                std::cout << "Solution Does Not Exist\n";
                break;
            }
            if (!ladder.empty()) ladder.pop_back();
            past_size = ladder.size();
        }
        ++attempts;
    }
    if (attempts == max_attempts) {
        std::cerr << "Error: Exceeded maximum attempts, possible infinite loop.\n";
        return 1;
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
    if (word == wordFinal) {
        return true;
    }

    size_t wordLength = word.length();

    for (const auto& strcomp : words) {
        if (isRedundant(strcomp, ladder)) {
            continue;
        }
        // CWE-120: Ensure safe access
        if (strcomp.length() != wordLength) continue;

        int counter = 0;
        for (size_t i = 0; i < wordLength; i++) {
            if (word[i] == strcomp[i])
                counter++;
        }

        if (counter == static_cast<int>(wordLength) - 1) {
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
    for (const auto& w : ladder) {
        if (word == w) {
            return true;
        }
    }
    return false;
}