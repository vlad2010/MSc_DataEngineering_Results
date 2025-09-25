#include <iostream>
#include <fstream>

int main() {
    // Open SVG file
    std::ofstream svgFile("output.svg");

    // Check if file is open
    if (!svgFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    // Variables for SVG dimensions and rectangle dimensions
    int svgWidth = 200;
    int svgHeight = 200;
    int rectWidth = 120;
    int rectHeight = 80;

    // Write SVG content using variables
    svgFile << "<svg width=\"" << svgWidth << "\" height=\"" << svgHeight << "\" xmlns=\"http://www.w3.org/2000/svg\">" << std::endl;
    svgFile << "<rect width=\"" << rectWidth << "\" height=\"" << rectHeight << "\" fill=\"blue\"/>" << std::endl;
    svgFile << "</svg>" << std::endl;

    // Close the file
    svgFile.close();

    return 0;
}