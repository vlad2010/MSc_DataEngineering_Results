#include <memory>
#include <iostream>
#include <new> // For std::bad_alloc

size_t arraySize = 10;
std::unique_ptr<int[]> dynamicArray;

try {
    dynamicArray = std::make_unique<int[]>(arraySize);
    // Use dynamicArray as needed, e.g., dynamicArray[0] = 42;
} catch (const std::bad_alloc& e) {
    std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    // Handle error appropriately
}