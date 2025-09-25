#include <iostream>
#include <cstdint>

int main() {
    std::cout << "Size of size_t: " << sizeof(size_t) << " bytes or "
              << (sizeof(size_t) * 8) << " bits" << std::endl;

    std::cout << "Size of uint8_t: " << sizeof(uint8_t) << " bytes or "
              << (sizeof(uint8_t) * 8) << " bits" << std::endl;

    std::cout << "Size of uint16_t: " << sizeof(uint16_t) << " bytes or "
              << (sizeof(uint16_t) * 8) << " bits" << std::endl;

    std::cout << "Size of uint32_t: " << sizeof(uint32_t) << " bytes or "
              << (sizeof(uint32_t) * 8) << " bits" << std::endl;

    std::cout << "Size of uint64_t: " << sizeof(uint64_t) << " bytes or "
              << (sizeof(uint64_t) * 8) << " bits" << std::endl;

    return 0;
}