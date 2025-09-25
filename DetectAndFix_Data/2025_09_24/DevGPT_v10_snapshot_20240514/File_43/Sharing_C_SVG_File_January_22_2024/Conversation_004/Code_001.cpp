#include <iostream>
#include <fstream>

using namespace std;  // Add this line to use the std namespace

int main() {
    // Open SVG file
    ofstream svgFile("output.svg");

    // Check if file is open
    if (!svgFile.is_open()) {
        cerr << "Error opening file!" << endl;
        return 1;
    }

    // Variables for SVG dimensions and rectangle dimensions
    int svgWidth = 200;
    int svgHeight = 200;
    int rectWidth = 120;
    int rectHeight = 80;

    // Write SVG content using variables
    svgFile << "<svg width=\"" << svgWidth << "\" height=\"" << svgHeight << "\" xmlns=\"http://www.w3.org/2000/svg\">" << endl;
    svgFile << "<rect width=\"" << rectWidth << "\" height=\"" << rectHeight << "\" fill=\"blue\"/>" << endl;
    svgFile << "</svg>" << endl;

    // Close the file
    svgFile.close();

    return 0;
}