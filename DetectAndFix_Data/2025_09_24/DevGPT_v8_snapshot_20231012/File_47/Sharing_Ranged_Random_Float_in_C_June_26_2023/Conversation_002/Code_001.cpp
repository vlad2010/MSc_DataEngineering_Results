#include <iostream>
#include <cmath>

struct Point {
    float x;
    float y;
};

float calculateDistance(const Point& p1, const Point& p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    return std::sqrt(dx * dx + dy * dy);
}

int main() {
    // Example usage
    Point point1 = {1.0, 2.0};
    Point point2 = {4.0, 6.0};

    float distance = calculateDistance(point1, point2);
    std::cout << "Distance between point1 and point2: " << distance << std::endl;

    return 0;
}
