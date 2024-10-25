#include <iostream>
#include <cmath>

struct Vector2D {
    float x;
    float y;

    Vector2D(float xVal, float yVal) : x(xVal), y(yVal) {}
};

void Normalize(Vector2D& vector) {
    float length = std::sqrt(vector.x * vector.x + vector.y * vector.y);
    vector.x /= length;
    vector.y /= length;
}

void Reflect(Vector2D& velocity, const Vector2D& collisionNormal) {
    float dotProduct = velocity.x * collisionNormal.x + velocity.y * collisionNormal.y;
    velocity.x = velocity.x - 2.0f * dotProduct * collisionNormal.x;
    velocity.y = velocity.y - 2.0f * dotProduct * collisionNormal.y;
}

int main() {
    Vector2D object1Velocity(1.0f, 1.0f);
    Vector2D object2Velocity(-1.0f, -1.0f);

    Vector2D collisionDirection(-1.0f, -1.0f);  // Assuming collision from top-left to bottom-right

    // Calculate collision normal
    Vector2D collisionNormal = collisionDirection;
    Normalize(collisionNormal);

    Reflect(object1Velocity, collisionNormal);
    Reflect(object2Velocity, collisionNormal);

    std::cout << "New velocities:" << std::endl;
    std::cout << "Object 1: (" << object1Velocity.x << ", " << object1Velocity.y << ")" << std::endl;
    std::cout << "Object 2: (" << object2Velocity.x << ", " << object2Velocity.y << ")" << std::endl;

    return 0;
}
