#include <iostream>
#include <cmath>

struct Vector2D {
    float x;
    float y;

    Vector2D(float xVal, float yVal) : x(xVal), y(yVal) {}
};

Vector2D CalculateCollisionNormal(const Vector2D& collisionDirection) {
    // Assuming collisionDirection is a normalized vector representing the direction of collision

    // Calculate the perpendicular vector (collision normal)
    return Vector2D(-collisionDirection.y, collisionDirection.x);
}

void Reflect(Vector2D& velocity, const Vector2D& collisionNormal) {
    float dotProduct = velocity.x * collisionNormal.x + velocity.y * collisionNormal.y;
    velocity.x = velocity.x - 2.0f * dotProduct * collisionNormal.x;
    velocity.y = velocity.y - 2.0f * dotProduct * collisionNormal.y;
}

int main() {
    Vector2D object1Velocity(1.0f, 1.0f);
    Vector2D object2Velocity(-1.0f, -1.0f);

    Vector2D collisionDirection(-1.0f, 0.0f);  // Assuming collision from left to right

    Vector2D collisionNormal = CalculateCollisionNormal(collisionDirection);

    Reflect(object1Velocity, collisionNormal);
    Reflect(object2Velocity, collisionNormal);

    std::cout << "New velocities:" << std::endl;
    std::cout << "Object 1: (" << object1Velocity.x << ", " << object1Velocity.y << ")" << std::endl;
    std::cout << "Object 2: (" << object2Velocity.x << ", " << object2Velocity.y << ")" << std::endl;

    return 0;
}
