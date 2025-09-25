#include <vector>
#include <memory>
#include <array>
#include <stdexcept>
#include <algorithm>

// Solution 1: Using std::vector (RECOMMENDED - Most flexible and safe)
void safeApproach1() {
    const size_t arraySize = 10;
    
    // Automatically manages memory, provides bounds checking with at()
    std::vector<int> dynamicArray(arraySize, 0);  // Initialize all elements to 0
    
    // Safe access with bounds checking
    try {
        for (size_t i = 0; i < arraySize; ++i) {
            dynamicArray.at(i) = static_cast<int>(i);  // at() throws if out of bounds
        }
    } catch (const std::out_of_range& e) {
        // Handle out of bounds access
    }
    
    // Memory automatically freed when vector goes out of scope
}

// Solution 2: Using std::unique_ptr for array (if you must use dynamic allocation)
void safeApproach2() {
    const size_t arraySize = 10;
    
    // Smart pointer automatically manages memory
    std::unique_ptr<int[]> dynamicArray = std::make_unique<int[]>(arraySize);
    
    // Initialize the array (make_unique doesn't initialize for arrays)
    std::fill_n(dynamicArray.get(), arraySize, 0);
    
    // Use with manual bounds checking
    for (size_t i = 0; i < arraySize; ++i) {
        if (i < arraySize) {  // Explicit bounds check
            dynamicArray[i] = static_cast<int>(i);
        }
    }
    
    // Memory automatically freed when unique_ptr goes out of scope
}

// Solution 3: If size is known at compile time, use std::array
template<size_t N = 10>
void safeApproach3() {
    // Stack allocation, no dynamic memory needed
    std::array<int, N> staticArray{};  // Zero-initialized
    
    // Safe access with bounds checking
    for (size_t i = 0; i < staticArray.size(); ++i) {
        staticArray.at(i) = static_cast<int>(i);
    }
    
    // No manual memory management needed
}

// Solution 4: If you MUST use raw pointers (NOT RECOMMENDED)
void improvedRawPointerApproach() {
    const size_t arraySize = 10;
    int* dynamicArray = nullptr;
    
    try {
        // Allocate and immediately initialize
        dynamicArray = new int[arraySize]{};  // Zero-initialization
        
        // Use the array with bounds checking
        for (size_t i = 0; i < arraySize; ++i) {
            dynamicArray[i] = static_cast<int>(i);
        }
        
        // ... use the array ...
        
        // Clean up
        delete[] dynamicArray;
        dynamicArray = nullptr;  // Avoid dangling pointer
        
    } catch (const std::bad_alloc& e) {
        // Handle allocation failure
        if (dynamicArray) {
            delete[] dynamicArray;
            dynamicArray = nullptr;
        }
        throw;  // Re-throw or handle as appropriate
    }
}

// RECOMMENDED PRODUCTION CODE:
class SafeArrayWrapper {
private:
    std::vector<int> data;
    
public:
    explicit SafeArrayWrapper(size_t size) 
        : data(size, 0) {  // Initialize to 0
        if (size == 0) {
            throw std::invalid_argument("Array size must be greater than 0");
        }
    }
    
    // Safe element access
    int& at(size_t index) {
        return data.at(index);  // Throws std::out_of_range if index is invalid
    }
    
    const int& at(size_t index) const {
        return data.at(index);
    }
    
    size_t size() const noexcept {
        return data.size();
    }
    
    // Iterator support for range-based for loops
    auto begin() noexcept { return data.begin(); }
    auto end() noexcept { return data.end(); }
    auto begin() const noexcept { return data.begin(); }
    auto end() const noexcept { return data.end(); }
};

// Usage example
int main() {
    try {
        SafeArrayWrapper array(10);
        
        // Safe usage
        for (size_t i = 0; i < array.size(); ++i) {
            array.at(i) = static_cast<int>(i * 2);
        }
        
        // Range-based for loop
        for (const auto& value : array) {
            // Process value safely
        }
        
    } catch (const std::exception& e) {
        // Handle any exceptions
    }
    
    return 0;
}