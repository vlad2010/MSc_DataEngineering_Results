#ifndef SVG_WRITER_H
#define SVG_WRITER_H

#include <iostream>
#include <fstream>

class SVGWriter {
public:
    SVGWriter(const std::string &filename, int width, int height);
    ~SVGWriter();

    void writeRect(int width, int height, const std::string &fillColor);
    void writeCircle(double cx, double cy, double r, const std::string &strokeColor, const std::string &fillColor, double strokeWidth);
    void writeMarker(const std::string &markerId);
    void writeLine(double x1, double y1, double x2, double y2, const std::string &strokeColor, double strokeWidth, const std::string &markerStart);

private:
    std::ofstream svgFile;
};

#endif // SVG_WRITER_H