struct FourBitInteger {
    unsigned char value : 4; // 4-bit bit-field
};

int main() {
    FourBitInteger myFourBit;
    myFourBit.value = 0b1010; // Assigning a 4-bit value

    // You can use myFourBit.value as a 4-bit unsigned integer
    return 0;
}