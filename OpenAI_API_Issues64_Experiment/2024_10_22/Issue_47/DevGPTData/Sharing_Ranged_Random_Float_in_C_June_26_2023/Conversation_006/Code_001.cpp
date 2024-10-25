#include <iostream>

struct Point {
    int x;
    int y;
};

int main() {
    Point p1 = {1, 2};
    Point p2 = {3, 4};
    Point p3 = {1, 2};

    if (p1 == p2) {
        std::cout << "p1 and p2 are equal" << std::endl;
    } else {
        std::cout << "p1 and p2 are not equal" << std::endl;
    }

    if (p1 == p3) {
        std::cout << "p1 and p3 are equal" << std::endl;
    } else {
        std::cout << "p1 and p3 are not equal" << std::endl;
    }

    return 0;
}
