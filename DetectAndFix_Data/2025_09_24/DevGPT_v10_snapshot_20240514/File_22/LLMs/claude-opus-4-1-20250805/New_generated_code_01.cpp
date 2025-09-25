#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <ctime>
#include <limits>
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

// Forward declaration to fix compilation issue
void outputBoard(string board[]);

bool checkCellChoice(int choice) {
    if (cin.fail() || choice < 1 || choice > NINE) {
        cout << "Invalid input! Please enter a number between 1 and 9." << endl;
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        return false;
    }
    return true;
}

bool validateCell(players player, string board[], int choice, bool bot, bool debugMode) {
    // Bounds checking for both bot and human
    if (bot) {
        // For bot, choice is 0-based index
        if (choice < 0 || choice >= NINE) {
            if (debugMode) {
                cout << "\033[31mInvalid cell index: " << choice << "\n\033[30m";
            }
            return false;
        }
        if (board[choice] != to_string(choice + 1)) {
            if (debugMode) {
                cout << "\033[31mCell is already selected!\n\033[30m";
            }
            return false;
        }
    } else {
        // For human, choice is 1-based
        if (choice < 1 || choice > NINE) {
            return false;
        }
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

void selectCell(players player, string board[], bool bot, bool debugMode) {
    int choice = 0;
    int x = 0, y = 0;
    
    if (!bot) { // If it's a player
        cout << "Player " + player.symbol + ", Make a selection: \n";
        if (debugMode) {
            cout << "\033[31mPlay function called.\n\033[30m";
        }
        
        // Input validation loop
        do {
            cin >> choice;
        } while (!checkCellChoice(choice) || !validateCell(player, board, choice, bot, debugMode));
        
        // Safe array access with bounds checking
        if (choice >= 1 && choice <= NINE) {
            board[choice - 1] = player.symbol; // Sets cell as taken
        }
    } else { // If it's the computer
        // Initialize random seed once (should be done in main, but adding here for safety)
        static bool seeded = false;
        if (!seeded) {
            srand(static_cast<unsigned int>(time(nullptr)));
            seeded = true;
        }
        
        // Find valid cell with proper bounds checking
        int attempts = 0;
        const int MAX_ATTEMPTS = 100; // Prevent infinite loop
        
        do {
            x = rand() % 3;
            y = rand() % 3;
            choice = x * 3 + y; // Calculate the cell index (0-8)
            
            if (debugMode) {
                cout << "\033[31mTrying row: " << x << " , col: " << y 
                     << " (index: " << choice << ")\033[30m\n";
            }
            
            attempts++;
            if (attempts >= MAX_ATTEMPTS) {
                cout << "Error: Unable to find valid cell for bot" << endl;
                return;
            }
        } while (!validateCell(player, board, choice, bot, debugMode));
        
        if (debugMode) {
            cout << "\033[31mPlaying in row: " << x << ", col: " << y 
                 << " (index: " << choice << ")\033[30m\n";
        }
        
        // Safe array access with final bounds check
        if (choice >= 0 && choice < NINE) {
            board[choice] = player.symbol;
        }
    }
}

// Additional security improvements for main function
void initializeBoard(string board[]) {
    for (int i = 0; i < NINE; i++) {
        board[i] = to_string(i + 1);
    }
}

// Example main function showing proper initialization
int main() {
    // Seed random number generator once at program start
    srand(static_cast<unsigned int>(time(nullptr)));
    
    string board[NINE];
    initializeBoard(board);
    
    players player1, player2;
    player1.symbol = "X";
    player2.symbol = "O";
    
    bool debugMode = false;
    
    // Example usage
    outputBoard(board);
    selectCell(player1, board, false, debugMode); // Human player
    outputBoard(board);
    selectCell(player2, board, true, debugMode);  // Bot player
    outputBoard(board);
    
    return 0;
}