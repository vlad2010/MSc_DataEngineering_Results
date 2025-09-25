#include <iostream>
#include <iomanip>
#include <string>
#include <limits>
#include <array>
#include <vector>
#include <random>

constexpr int NINE = 9; // Number of cells in the board

struct players {
    char symbol; // 'X' or 'O'
    int score;   // How many wins a player has
    players() : symbol(' '), score(0) {}
};

static inline bool isOccupied(const std::string& cell) {
    return cell == "X" || cell == "O";
}

void outputBoard(const std::array<std::string, NINE>& board) {
    std::cout << " -----------\n| ";
    for (int i = 0; i < NINE; i++) {
        if (i % 3 == 0 && i != 0) {
            std::cout << "\n -----------\n| ";
        }
        std::cout << board[i] << " | ";
    }
    std::cout << "\n -----------\n";
}

// Validate a cell selection.
// For human (!bot): choice is expected to be 1..9 (user input); function converts to 0-based safely.
// For bot (bot): choice is expected to be 0..8.
bool validateCell(const players& player,
                  const std::array<std::string, NINE>& board,
                  int choice,
                  bool bot,
                  bool debugMode) {
    if (!bot) {
        // Convert 1..9 to 0..8 with full bounds checks
        if (choice < 1 || choice > NINE) {
            if (debugMode) {
                std::cout << "\033[31mHuman choice out of range.\033[30m\n";
            }
            return false;
        }
        const int idx = choice - 1;
        // The cell is free if it still holds its original number as a string
        if (board[idx] != std::to_string(choice)) {
            std::cout << "This slot is already selected" << std::endl;
            if (debugMode) {
                std::cout << "\033[31mChecking the board for a winning sequence\n\033[30m";
                outputBoard(board);
                std::cout << "Player " << player.symbol << ", Make a selection: \n";
                std::cout << "\033[31mPlay function called.\n\033[30m";
            }
            return false;
        }
    } else {
        // Bot path: choice must already be 0..8
        if (choice < 0 || choice >= NINE) {
            if (debugMode) {
                std::cout << "\033[31mBot choice out of range.\033[30m\n";
            }
            return false;
        }
        const int idx = choice;
        // A cell is occupied if it's X or O, regardless of which player is moving
        if (isOccupied(board[idx])) {
            if (debugMode) {
                std::cout << "\033[31mCell is already selected!\nChanging selection.\n\033[30m\n";
            }
            return false;
        }
    }
    return true;
}

// Validates that human input is between 1..9 only (no I/O side effects here).
bool checkCellChoice(int choice) {
    return choice >= 1 && choice <= NINE;
}

void selectCell(const players& player,
                std::array<std::string, NINE>& board,
                int& choice,
                bool bot,
                bool debugMode) {
    if (!bot) { // Human
        std::cout << "Player " << player.symbol << ", Make a selection: \n";
        if (debugMode) {
            std::cout << "\033[31mPlay function called.\n\033[30m";
        }

        // Robust input loop with EOF handling
        while (true) {
            if (!(std::cin >> choice)) {
                if (std::cin.eof()) {
                    std::cout << "Input closed. Aborting move.\n";
                    return; // Avoid infinite loop on EOF (CWE-835)
                }
                std::cout << "Invalid input! Please enter a number between 1 and 9." << std::endl;
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                continue;
            }
            if (!checkCellChoice(choice) || !validateCell(player, board, choice, /*bot=*/false, debugMode)) {
                std::cout << "Invalid input! Please enter a number between 1 and 9." << std::endl;
                // Clear any trailing junk on the line
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                continue;
            }
            break;
        }
        // Enforce only 'X' or 'O' written to board (CWE-150/CWE-20 mitigation)
        const std::string sym = (player.symbol == 'X' ? "X" : (player.symbol == 'O' ? "O" : "X"));
        board[choice - 1] = sym;
    } else { // Bot
        // Build a list of free cells to avoid infinite loop when board is full (CWE-835)
        std::vector<int> freeCells;
        freeCells.reserve(NINE);
        for (int i = 0; i < NINE; ++i) {
            if (!isOccupied(board[i])) {
                freeCells.push_back(i);
            }
        }

        if (freeCells.empty()) {
            if (debugMode) {
                std::cout << "\033[31mNo free cells remaining. Bot cannot move.\033[30m\n";
            }
            return;
        }

        // Use a better RNG (not security-critical here, but preferable to rand())
        static std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, static_cast<int>(freeCells.size()) - 1);
        int idx = freeCells[dist(rng)];
        int x = idx / 3;
        int y = idx % 3;

        if (debugMode) {
            std::cout << "\033[31mrow: " << x << " , col: " << y << "\033[30m\n";
        }

        // Final validation (defensive)
        if (!validateCell(player, board, idx, /*bot=*/true, debugMode)) {
            if (debugMode) {
                std::cout << "\033[31mUnexpected: chosen cell invalid.\033[30m\n";
            }
            return;
        }

        if (debugMode) {
            std::cout << "\033[31mPlaying in row: " << x << ", col: " << y << "\033[30m\n";
        }

        const std::string sym = (player.symbol == 'X' ? "X" : (player.symbol == 'O' ? "O" : "X"));
        board[idx] = sym;
        choice = idx; // keep caller informed if they rely on choice
    }
}