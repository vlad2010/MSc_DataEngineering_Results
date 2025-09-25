#ifndef SVG_WRITER_H
#define SVG_WRITER_H

#include <iostream>
#include <fstream>

void openSVGFile(std::ofstream &svgFile, const std::string &filename, int width, int height);

void writeRect(std::ofstream &svgFile, int width, int height, const std::string &fillColor);

void writeCircle(std::ofstream &svgFile, double cx, double cy, double r, const std::string &strokeColor, const std::string &fillColor, double strokeWidth);

void closeSVGFile(std::ofstream &svgFile);

#endif // SVG_WRITER_H