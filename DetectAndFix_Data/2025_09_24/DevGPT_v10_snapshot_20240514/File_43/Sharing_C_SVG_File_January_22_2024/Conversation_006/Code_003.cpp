#include "svg_writer.h"

int main() {
    // Variables for SVG dimensions and rectangle dimensions
    int svgWidth = 200;
    int svgHeight = 200;
    int rectWidth = 120;
    int rectHeight = 80;

    // Open SVG file
    std::ofstream svgFile;
    openSVGFile(svgFile, "output.svg", svgWidth, svgHeight);

    // Write SVG content using variables
    writeRect(svgFile, rectWidth, rectHeight, "blue");

    // Close the file
    closeSVGFile(svgFile);

    return 0;
}