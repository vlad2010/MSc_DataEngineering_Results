#include <iostream>
#include <fstream>

// Hypothetical Point2D class
class Point2D {
public:
    double x;
    double y;

    Point2D(double x_val, double y_val) : x(x_val), y(y_val) {}
};

void draw_vector(std::ofstream &svgFile, Point2D &tail, Point2D &head, const std::string &stroke_color) {
    svgFile << "<line x1=\"" << head.x << "\" x2=\"" << tail.x << "\" y1=\"" << head.y << "\" y2=\"" << tail.y << "\" stroke=\"" << stroke_color << "\" stroke-width=\"5\" marker-start=\"url(#Arrow1Sstart)\" />" << std::endl;
}

int main() {
    // Example usage
    std::ofstream svgFile("output.svg");

    if (!svgFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    Point2D tail(100.0, 200.0);
    Point2D head(300.0, 400.0);
    std::string strokeColor = "blue";

    draw_vector(svgFile, tail, head, strokeColor);

    svgFile.close();

    return 0;
}