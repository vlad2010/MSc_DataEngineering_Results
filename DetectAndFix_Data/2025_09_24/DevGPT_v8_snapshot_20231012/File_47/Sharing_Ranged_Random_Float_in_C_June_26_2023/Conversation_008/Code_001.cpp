#include <iostream>
#include <SDL.h>

int main() {
    SDL_version compiled;
    SDL_version linked;

    SDL_VERSION(&compiled);
    SDL_GetVersion(&linked);

    std::cout << "SDL compiled version: " << static_cast<int>(compiled.major) << "."
              << static_cast<int>(compiled.minor) << "." << static_cast<int>(compiled.patch) << std::endl;
    std::cout << "SDL linked version: " << static_cast<int>(linked.major) << "."
              << static_cast<int>(linked.minor) << "." << static_cast<int>(linked.patch) << std::endl;

    return 0;
}
