#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>

using namespace std;

int main() {
    int cases = 0, partyAmount = 0, voteAmount = 0;
    string filename, party, candidate, vote, winner;

    cout << "Enter filename: ";
    cin >> filename;
    ifstream infile(filename);

    // CWE-252: Check if file opened successfully
    if (!infile.is_open()) {
        cerr << "Error: Could not open file '" << filename << "'." << endl;
        return 1;
    }

    // CWE-20: Check input extraction
    if (!(infile >> cases)) {
        cerr << "Error: Failed to read number of cases." << endl;
        return 1;
    }

    for (int i = 0; i < cases; i++) {
        unordered_map<string, int> votes;
        unordered_map<string, string> parties;
        bool tie = false;
        int maxVotes = 0; // CWE-457: Reset for each case
        winner = "";

        // CWE-20: Check input extraction
        if (!(infile >> partyAmount)) {
            cerr << "Error: Failed to read party amount for case " << i + 1 << "." << endl;
            return 1;
        }
        infile.ignore();

        for (int j = 0; j < partyAmount; j++) {
            if (!getline(infile, candidate) || !getline(infile, party)) {
                cerr << "Error: Failed to read candidate or party for case " << i + 1 << "." << endl;
                return 1;
            }
            parties[candidate] = party;
        }

        if (!(infile >> voteAmount)) {
            cerr << "Error: Failed to read vote amount for case " << i + 1 << "." << endl;
            return 1;
        }
        infile.ignore();

        for (int j = 0; j < voteAmount; j++) {
            if (!getline(infile, vote)) {
                cerr << "Error: Failed to read vote for case " << i + 1 << "." << endl;
                return 1;
            }
            votes[vote]++;
        }

        // Find the winner or check for a tie
        for (const auto& [cand, voteCount] : votes) {
            // CWE-704: Check if candidate exists in parties map
            auto it = parties.find(cand);
            if (it == parties.end()) {
                cerr << "Warning: Vote for unknown candidate '" << cand << "' in case " << i + 1 << "." << endl;
                continue;
            }
            if (voteCount > maxVotes) {
                winner = it->second;
                maxVotes = voteCount;
                tie = false;
            } else if (voteCount == maxVotes && voteCount != 0) {
                tie = true;
            }
        }

        // Display results
        cout << "Case " << i + 1 << " results: ";
        if (tie || winner.empty()) {
            cout << "Tie" << endl;
        } else {
            cout << winner << endl;
        }
    }

    infile.close(); // CWE-772: Explicitly close file
    return 0;
}