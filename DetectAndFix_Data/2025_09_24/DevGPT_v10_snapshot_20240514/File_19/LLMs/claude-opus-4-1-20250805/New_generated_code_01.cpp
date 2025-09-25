#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <limits>
#include <filesystem>

using namespace std;

// Security constants
const int MAX_CASES = 1000;
const int MAX_PARTIES = 10000;
const int MAX_VOTES = 100000;
const int MAX_STRING_LENGTH = 1000;
const int MAX_FILENAME_LENGTH = 255;

bool isValidFilename(const string& filename) {
    // Check filename length
    if (filename.empty() || filename.length() > MAX_FILENAME_LENGTH) {
        return false;
    }
    
    // Basic path traversal prevention - disallow .. and absolute paths
    if (filename.find("..") != string::npos || 
        filename.find("/") == 0 || 
        filename.find("\\") == 0 ||
        (filename.length() > 1 && filename[1] == ':')) {
        return false;
    }
    
    return true;
}

bool safeGetline(ifstream& infile, string& str) {
    if (!getline(infile, str)) {
        return false;
    }
    // Limit string length to prevent memory exhaustion
    if (str.length() > MAX_STRING_LENGTH) {
        str = str.substr(0, MAX_STRING_LENGTH);
    }
    return true;
}

int main() {
    try {
        // Variable declarations
        int cases = 0, partyAmount = 0, voteAmount = 0;
        string filename, party, candidate, vote, winner;
        
        // File handling with validation
        cout << "Enter filename: ";
        if (!getline(cin, filename)) {
            cerr << "Error: Failed to read filename" << endl;
            return 1;
        }
        
        // Validate filename
        if (!isValidFilename(filename)) {
            cerr << "Error: Invalid filename" << endl;
            return 1;
        }
        
        ifstream infile(filename);
        if (!infile.is_open()) {
            cerr << "Error: Cannot open file '" << filename << "'" << endl;
            return 1;
        }
        
        // Read and validate number of cases
        if (!(infile >> cases)) {
            cerr << "Error: Invalid input format for cases" << endl;
            return 1;
        }
        
        if (cases < 0 || cases > MAX_CASES) {
            cerr << "Error: Invalid number of cases (must be between 0 and " << MAX_CASES << ")" << endl;
            return 1;
        }

        for (int i = 0; i < cases; i++) {
            // Reset variables for each case - FIX for CWE-457
            unordered_map<string, int> votes;
            unordered_map<string, string> parties;
            int maxVotes = 0;  // Reset maxVotes for each case
            bool tie = false;
            winner.clear();

            // Read and validate party information
            if (!(infile >> partyAmount)) {
                cerr << "Error: Invalid input format for party amount in case " << (i+1) << endl;
                return 1;
            }
            
            if (partyAmount < 0 || partyAmount > MAX_PARTIES) {
                cerr << "Error: Invalid number of parties in case " << (i+1) << endl;
                return 1;
            }
            
            infile.ignore(numeric_limits<streamsize>::max(), '\n');
            
            for (int j = 0; j < partyAmount; j++) {
                if (!safeGetline(infile, candidate) || !safeGetline(infile, party)) {
                    cerr << "Error: Unexpected end of file while reading parties in case " << (i+1) << endl;
                    return 1;
                }
                
                // Validate non-empty strings
                if (candidate.empty() || party.empty()) {
                    cerr << "Error: Empty candidate or party name in case " << (i+1) << endl;
                    return 1;
                }
                
                parties[candidate] = party;
            }

            // Read and validate vote information
            if (!(infile >> voteAmount)) {
                cerr << "Error: Invalid input format for vote amount in case " << (i+1) << endl;
                return 1;
            }
            
            if (voteAmount < 0 || voteAmount > MAX_VOTES) {
                cerr << "Error: Invalid number of votes in case " << (i+1) << endl;
                return 1;
            }
            
            infile.ignore(numeric_limits<streamsize>::max(), '\n');
            
            for (int j = 0; j < voteAmount; j++) {
                if (!safeGetline(infile, vote)) {
                    cerr << "Error: Unexpected end of file while reading votes in case " << (i+1) << endl;
                    return 1;
                }
                
                // Only count votes for registered candidates
                if (parties.find(vote) != parties.end()) {
                    votes[vote]++;
                }
                // Invalid votes are silently ignored (as per typical election rules)
            }

            // Find the winner or check for a tie
            for (const auto& [candidateName, voteCount] : votes) {
                if (voteCount > maxVotes) {
                    // Safe access - we know candidate exists in parties
                    winner = parties[candidateName];
                    maxVotes = voteCount;
                    tie = false;
                } else if (voteCount == maxVotes && maxVotes > 0) {
                    tie = true;
                }
            }

            // Display results
            if (tie) {
                cout << "Case " << i + 1 << " results: Tie" << endl;
            } else if (!winner.empty()) {
                cout << "Case " << i + 1 << " results: " << winner << endl;
            } else {
                cout << "Case " << i + 1 << " results: No valid votes" << endl;
            }
        }
        
        infile.close();
        
    } catch (const bad_alloc& e) {
        cerr << "Error: Out of memory" << endl;
        return 1;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}