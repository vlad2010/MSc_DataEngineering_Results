#include <cstdint>

int main() {
    const uint8_t myConstUint8 = 42;

    // Compiler will generate an error if you try to modify myConstUint8
    // myConstUint8 = 10; // This line will result in a compilation error

    return 0;
}