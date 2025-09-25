#include <iostream>
#include <cstdint>

int main() {
    std::cout << "Size of size_t: " << sizeof(size_t) << " bytes" << std::endl;
    std::cout << "Size of uint8_t: " << sizeof(uint8_t) << " bytes" << std::endl;
    std::cout << "Size of uint16_t: " << sizeof(uint16_t) << " bytes" << std::endl;
    std::cout << "Size of uint32_t: " << sizeof(uint32_t) << " bytes" << std::endl;
    std::cout << "Size of uint64_t: " << sizeof(uint64_t) << " bytes" << std::endl;

    return 0;
}