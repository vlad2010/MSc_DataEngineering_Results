#include "snes_ppu.h"

int main() {
    using namespace SNES;

    // Example: Set screen brightness to maximum and enable display
    PPU::setINIDISP(0x0F, true);

    // Example: Set Background Mode 1
    PPU::setBGMODE(1);

    // Example: Set object size and address
    PPU::setOBSEL(0, 0x10);

    return 0;
}