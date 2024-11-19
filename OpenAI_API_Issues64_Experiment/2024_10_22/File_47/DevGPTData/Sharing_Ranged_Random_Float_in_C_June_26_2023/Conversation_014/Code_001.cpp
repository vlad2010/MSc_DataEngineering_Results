SDL_Rect rect1 = { 10, 20, 100, 100 };
SDL_Rect rect2 = { 10, 20, 100, 100 };
SDL_Rect rect3 = { 0, 0, 50, 50 };

if (rect1 == rect2) {
    // rect1 and rect2 are equal
    // Their x, y, w, and h fields have the same values
} else {
    // rect1 and rect2 are not equal
}

if (rect1 == rect3) {
    // rect1 and rect3 are equal
} else {
    // rect1 and rect3 are not equal
}
