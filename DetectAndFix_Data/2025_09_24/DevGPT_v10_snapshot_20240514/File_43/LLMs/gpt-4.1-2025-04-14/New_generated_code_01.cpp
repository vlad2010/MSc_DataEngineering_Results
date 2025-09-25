#include "svg_writer.h"
#include <stdexcept>
#include <sstream>
#include <iostream>

// Helper function to escape XML special characters
static std::string escapeXML(const std::string& input) {
    std::ostringstream oss;
    for (char c : input) {
        switch (c) {
            case '&':  oss << "&amp;";  break;
            case '<':  oss << "&lt;";   break;
            case '>':  oss << "&gt;";   break;
            case '"':  oss << "&quot;"; break;
            case '\'': oss << "&apos;"; break;
            default:   oss << c;        break;
        }
    }
    return oss.str();
}

SVGWriter::SVGWriter(const std::string &filename, int width, int height) {
    svgFile.open(filename);

    if (!svgFile.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    svgFile << "<svg width=\"" << width << "\" height=\"" << height << "\" xmlns=\"http://www.w3.org/2000/svg\">" << std::endl;
    if (!svgFile) throw std::runtime_error("Error writing to SVG file during header.");
}

SVGWriter::~SVGWriter() {
    if (svgFile.is_open()) {
        svgFile << "</svg>" << std::endl;
        svgFile.close();
    }
}

void SVGWriter::writeRect(int width, int height, const std::string &fillColor) {
    svgFile << "<rect width=\"" << width << "\" height=\"" << height
            << "\" fill=\"" << escapeXML(fillColor) << "\"/>" << std::endl;
    if (!svgFile) throw std::runtime_error("Error writing <rect> to SVG file.");
}

void SVGWriter::writeCircle(double cx, double cy, double r, const std::string &strokeColor, const std::string &fillColor, double strokeWidth) {
    svgFile << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << r
            << "\" stroke=\"" << escapeXML(strokeColor)
            << "\" fill=\"" << escapeXML(fillColor)
            << "\" stroke-width=\"" << strokeWidth << "\" />" << std::endl;
    if (!svgFile) throw std::runtime_error("Error writing <circle> to SVG file.");
}

void SVGWriter::writeMarker(const std::string &markerId) {
    svgFile << "<defs>" << std::endl;
    svgFile << "  <marker style=\"overflow:visible\" id=\"" << escapeXML(markerId)
            << "\" refX=\"0.0\" refY=\"0.0\" orient=\"auto\">" << std::endl;
    svgFile << "    <path transform=\"scale(0.2) translate(6,0)\" style=\"fill-rule:evenodd;fill:context-stroke;stroke:context-stroke;stroke-width:1.0pt\" d=\"M 0.0,0.0 L 5.0,-5.0 L -12.5,0.0 L 5.0,5.0 L 0.0,0.0 z \" />" << std::endl;
    svgFile << "  </marker>" << std::endl;
    svgFile << "</defs>" << std::endl;
    if (!svgFile) throw std::runtime_error("Error writing <marker> to SVG file.");
}

void SVGWriter::writeLine(double x1, double y1, double x2, double y2, const std::string &strokeColor, double strokeWidth, const std::string &markerStart) {
    svgFile << "<line x1=\"" << x1 << "\" y1=\"" << y1
            << "\" x2=\"" << x2 << "\" y2=\"" << y2
            << "\" stroke=\"" << escapeXML(strokeColor)
            << "\" stroke-width=\"" << strokeWidth
            << "\" marker-start=\"url(#" << escapeXML(markerStart) << ")\" />" << std::endl;
    if (!svgFile) throw std::runtime_error("Error writing <line> to SVG file.");
}