#include "svg_writer.h"

SVGWriter::SVGWriter(const std::string &filename, int width, int height) {
    svgFile.open(filename);

    if (!svgFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        exit(1);
    }

    svgFile << "<svg width=\"" << width << "\" height=\"" << height << "\" xmlns=\"http://www.w3.org/2000/svg\">" << std::endl;
}

SVGWriter::~SVGWriter() {
    svgFile << "</svg>" << std::endl;
    svgFile.close();
}

void SVGWriter::writeRect(int width, int height, const std::string &fillColor) {
    svgFile << "<rect width=\"" << width << "\" height=\"" << height << "\" fill=\"" << fillColor << "\"/>" << std::endl;
}

void SVGWriter::writeCircle(double cx, double cy, double r, const std::string &strokeColor, const std::string &fillColor, double strokeWidth) {
    svgFile << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << r << "\" stroke=\"" << strokeColor << "\" fill=\"" << fillColor << "\" stroke-width=\"" << strokeWidth << "\" />" << std::endl;
}

void SVGWriter::writeMarker(const std::string &markerId) {
    svgFile << "<defs>" << std::endl;
    svgFile << "  <marker style=\"overflow:visible\" id=\"" << markerId << "\" refX=\"0.0\" refY=\"0.0\" orient=\"auto\">" << std::endl;
    svgFile << "    <path transform=\"scale(0.2) translate(6,0)\" style=\"fill-rule:evenodd;fill:context-stroke;stroke:context-stroke;stroke-width:1.0pt\" d=\"M 0.0,0.0 L 5.0,-5.0 L -12.5,0.0 L 5.0,5.0 L 0.0,0.0 z \" />" << std::endl;
    svgFile << "  </marker>" << std::endl;
    svgFile << "</defs>" << std::endl;
}

void SVGWriter::writeLine(double x1, double y1, double x2, double y2, const std::string &strokeColor, double strokeWidth, const std::string &markerStart) {
    svgFile << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2 << "\" y2=\"" << y2 << "\" stroke=\"" << strokeColor << "\" stroke-width=\"" << strokeWidth << "\" marker-start=\"url(#" << markerStart << ")\" />" << std::endl;
}