#include "svg_writer.h"

int main() {
    // Variables for SVG dimensions and rectangle dimensions
    int svgWidth = 600;
    int svgHeight = 600;
    int rectWidth = 120;
    int rectHeight = 80;

    double circleCX = 504.2;
    double circleCY = 403.5;
    double circleRadius = 3;
    std::string circleStrokeColor = "purple";
    std::string circleFillColor = "purple";
    double circleStrokeWidth = 1;

    // Open SVG file
    std::ofstream svgFile;
    openSVGFile(svgFile, "output.svg", svgWidth, svgHeight);

    // Write SVG content using variables
    writeRect(svgFile, rectWidth, rectHeight, "blue");
    writeCircle(svgFile, circleCX, circleCY, circleRadius, circleStrokeColor, circleFillColor, circleStrokeWidth);

    // Close the file
    closeSVGFile(svgFile);

    return 0;
}