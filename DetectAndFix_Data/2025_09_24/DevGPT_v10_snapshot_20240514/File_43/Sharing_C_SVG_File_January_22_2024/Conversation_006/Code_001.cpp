#ifndef SVG_WRITER_H
#define SVG_WRITER_H

#include <iostream>
#include <fstream>

void openSVGFile(std::ofstream &svgFile, const std::string &filename, int width, int height);

void writeRect(std::ofstream &svgFile, int width, int height, const std::string &fillColor);

void closeSVGFile(std::ofstream &svgFile);

#endif // SVG_WRITER_H