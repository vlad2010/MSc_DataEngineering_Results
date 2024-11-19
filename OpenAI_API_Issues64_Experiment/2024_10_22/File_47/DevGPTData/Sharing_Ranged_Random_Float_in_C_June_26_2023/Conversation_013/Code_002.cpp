SDL_Rect rectA = { 0, 0, 100, 100 };
SDL_Rect rectB = { 50, 50, 100, 100 };
SDL_Rect intersection;

SDL_bool hasIntersection = SDL_IntersectRect(&rectA, &rectB, &intersection);

if (hasIntersection == SDL_TRUE) {
    // The rectangles intersect
    // The intersection coordinates are in the 'intersection' rectangle
    // You can access the intersection coordinates using intersection.x, intersection.y, intersection.w, and intersection.h
} else {
    // The rectangles do not intersect
}
