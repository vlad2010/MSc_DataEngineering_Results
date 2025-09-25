#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <limits>
#include <ctime>
using namespace std;

const int NINE = 9; // Number of cells in the board

struct players {
    string symbol; // Either X or O depending on player
    int score; // How many wins a player has
    players() {
        symbol = "";
        score = 0;
    }
};

// Fixed: Added bounds check for choice
bool checkCellChoice(int choice) {
    if (cin.fail() || choice < 1 || choice > NINE) {
        cout << "Invalid input! Please enter a number between 1 and 9." << endl;
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        return false;
    }
    return true;
}

// Fixed: Added bounds check for choice
bool validateCell(players player, string board[], int choice, bool bot, bool debugMode) {
    if (choice < 1 || choice > NINE) { // Defensive check
        cout << "Invalid cell selection." << endl;
        return false;
    }
    if (!bot) { // If it's not the computer
        if (board[choice - 1] != to_string(choice)) {
            cout << "This slot is already selected" << endl;
            if (debugMode) {
                cout << "\033[31mChecking the board for a winning sequence\n\033[30m";
                outputBoard(board);
                cout << "Player " + player.symbol + ", Make a selection: \n";
                cout << "\033[31mPlay function called.\n\033[30m";
            }
            return false;
        }
    }
    else { // If it's the computer
        // For bot, choice is 0-based, so check bounds
        if (choice < 0 || choice >= NINE) {
            if (debugMode) {
                cout << "\033[31mBot made an invalid selection!\n\033[30m\n";
            }
            return false;
        }
        if (board[choice] == player.symbol || board[choice] == "X" || board[choice] == "O") {
            if (debugMode) {
                cout << "\033[31mCell is already selected!\nChanging selection.\n\033[30m\n";
            }
            return false;
        }
    }
    return true;
}

void outputBoard(string board[]) {
    cout << " -----------\n| "; // Top of board
    for (int i = 0; i < NINE; i++) {
        if (i % 3 == 0 && i != 0) {
            cout << endl << " -----------\n| "; // line endings
        }
        cout << board[i] << " | "; // Outputs cells
    }
    cout << "\n -----------\n"; // Bottom of board
}

// Fixed: Pass choice by reference, validate input, and check bounds
void selectCell(players player, string board[], int &choice, bool bot, bool debugMode) {
    int x = 0, y = 0;
    if (!bot) { // If it's a player
        cout << "Player " + player.symbol + ", Make a selection: \n";
        if (debugMode) {
            cout << "\033[31mPlay function called.\n\033[30m";
        }
        while (true) {
            cin >> choice;
            if (!cin) {
                cout << "Invalid input! Please enter a number between 1 and 9." << endl;
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                continue;
            }
            if (checkCellChoice(choice) && validateCell(player, board, choice, bot, debugMode)) {
                break;
            }
        }
        board[choice - 1] = player.symbol; // Sets cell as taken
    }
    else { // If it's the computer
        do {
            x = rand() % 3;
            y = rand() % 3;
            choice = x * 3 + y; // Calculate the cell index (0-based)
            if (debugMode) {
                cout << "\033[31mrow: " << x << " , col: " << y << "\033[30m\n";
            }
        } while (!validateCell(player, board, choice + 1, false, debugMode)); // Pass as 1-based for validation
        if (debugMode) {
            cout << "\033[31mPlaying in row: " << x << ", col: " << y << "\033[30m\n";
        }
        board[choice] = player.symbol;
    }
}

int main() {
    // Example usage
    srand(static_cast<unsigned int>(time(nullptr))); // Seed random number generator

    string board[NINE];
    for (int i = 0; i < NINE; ++i) board[i] = to_string(i + 1);

    players player1;
    player1.symbol = "X";
    int choice = 0;
    bool bot = false;
    bool debugMode = true;

    outputBoard(board);
    selectCell(player1, board, choice, bot, debugMode);
    outputBoard(board);
}