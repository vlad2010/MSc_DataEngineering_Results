#include <cmath>
#include <SDL.h>

constexpr float FLOAT_EPSILON = 0.0001f;

bool AreFRectsEqual(const SDL_FRect& rect1, const SDL_FRect& rect2) {
    return (std::abs(rect1.x - rect2.x) < FLOAT_EPSILON) &&
           (std::abs(rect1.y - rect2.y) < FLOAT_EPSILON) &&
           (std::abs(rect1.w - rect2.w) < FLOAT_EPSILON) &&
           (std::abs(rect1.h - rect2.h) < FLOAT_EPSILON);
}

int main() {
    SDL_FRect rect1 = { 10.5f, 20.3f, 100.7f, 100.1f };
    SDL_FRect rect2 = { 10.6f, 20.4f, 100.8f, 100.2f };
    SDL_FRect rect3 = { 10.5f, 20.3f, 100.7f, 100.1f };

    if (AreFRectsEqual(rect1, rect2)) {
        // rect1 and rect2 are approximately equal within the tolerance
    } else {
        // rect1 and rect2 are not equal
    }

    if (AreFRectsEqual(rect1, rect3)) {
        // rect1 and rect3 are approximately equal within the tolerance
    } else {
        // rect1 and rect3 are not equal
    }

    return 0;
}
