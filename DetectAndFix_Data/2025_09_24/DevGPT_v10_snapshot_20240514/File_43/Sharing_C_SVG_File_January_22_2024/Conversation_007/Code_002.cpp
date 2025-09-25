#include "svg_writer.h"

void openSVGFile(std::ofstream &svgFile, const std::string &filename, int width, int height) {
    svgFile.open(filename);

    if (!svgFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        exit(1);
    }

    svgFile << "<svg width=\"" << width << "\" height=\"" << height << "\" xmlns=\"http://www.w3.org/2000/svg\">" << std::endl;
}

void writeRect(std::ofstream &svgFile, int width, int height, const std::string &fillColor) {
    svgFile << "<rect width=\"" << width << "\" height=\"" << height << "\" fill=\"" << fillColor << "\"/>" << std::endl;
}

void writeCircle(std::ofstream &svgFile, double cx, double cy, double r, const std::string &strokeColor, const std::string &fillColor, double strokeWidth) {
    svgFile << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << r << "\" stroke=\"" << strokeColor << "\" fill=\"" << fillColor << "\" stroke-width=\"" << strokeWidth << "\" />" << std::endl;
}

void closeSVGFile(std::ofstream &svgFile) {
    svgFile << "</svg>" << std::endl;
    svgFile.close();
}