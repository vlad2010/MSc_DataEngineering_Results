#include <iostream>
#include <cmath>

struct Vector2D {
    float x;
    float y;

    Vector2D(float xVal, float yVal) : x(xVal), y(yVal) {}
};

void Reflect(Vector2D& velocity, const Vector2D& collisionNormal) {
    float dotProduct = velocity.x * collisionNormal.x + velocity.y * collisionNormal.y;
    velocity.x = velocity.x - 2.0f * dotProduct * collisionNormal.x;
    velocity.y = velocity.y - 2.0f * dotProduct * collisionNormal.y;
}

int main() {
    Vector2D object1Velocity(1.0f, 1.0f);
    Vector2D object2Velocity(-1.0f, -1.0f);

    Vector2D collisionDirection(-1.0f, 0.0f);  // Assuming collision from left to right

    if (collisionDirection.x < 0.0f) {
        Reflect(object1Velocity, Vector2D(-1.0f, 0.0f));  // Reflect object1's velocity along the x-axis
        Reflect(object2Velocity, Vector2D(-1.0f, 0.0f));  // Reflect object2's velocity along the x-axis
    } else if (collisionDirection.x > 0.0f) {
        Reflect(object1Velocity, Vector2D(1.0f, 0.0f));  // Reflect object1's velocity along the x-axis
        Reflect(object2Velocity, Vector2D(1.0f, 0.0f));  // Reflect object2's velocity along the x-axis
    } else if (collisionDirection.y < 0.0f) {
        Reflect(object1Velocity, Vector2D(0.0f, -1.0f));  // Reflect object1's velocity along the y-axis
        Reflect(object2Velocity, Vector2D(0.0f, -1.0f));  // Reflect object2's velocity along the y-axis
    } else if (collisionDirection.y > 0.0f) {
        Reflect(object1Velocity, Vector2D(0.0f, 1.0f));  // Reflect object1's velocity along the y-axis
        Reflect(object2Velocity, Vector2D(0.0f, 1.0f));  // Reflect object2's velocity along the y-axis
    }

    std::cout << "New velocities:" << std::endl;
    std::cout << "Object 1: (" << object1Velocity.x << ", " << object1Velocity.y << ")" << std::endl;
    std::cout << "Object 2: (" << object2Velocity.x << ", " << object2Velocity.y << ")" << std::endl;

    return 0;
}
