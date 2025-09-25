// snes_ppu.h
#pragma once

#include <cstdint>

namespace SNES {

    class PPU {
    public:
        // Control and Status Registers
        static void writeINIDISP(uint8_t value);
        static void writeOBSEL(uint8_t value);
        static void writeOAMADDL(uint8_t value);
        static void writeOAMADDH(uint8_t value);
        static void writeOAMDATA(uint8_t value);
        static void writeBGMODE(uint8_t value);
        static void writeMOSAIC(uint8_t value);
        static void writeBG1SC(uint8_t value);
        static void writeBG2SC(uint8_t value);
        static void writeBG3SC(uint8_t value);
        static void writeBG4SC(uint8_t value);
        static void writeBG12NBA(uint8_t value);
        static void writeBG34NBA(uint8_t value);
        static void writeBG1HOFS(uint8_t value);
        static void writeBG1VOFS(uint8_t value);
        static void writeBG2HOFS(uint8_t value);
        static void writeBG2VOFS(uint8_t value);
        static void writeBG3HOFS(uint8_t value);
        static void writeBG3VOFS(uint8_t value);
        static void writeBG4HOFS(uint8_t value);
        static void writeBG4VOFS(uint8_t value);
        static void writeVMAIN(uint8_t value);
        static void writeVMADDL(uint8_t value);
        static void writeVMADDH(uint8_t value);
        static void writeVMDATA(uint8_t value);
        static void writeM7HOFS(uint8_t value);
        static void writeM7VOFS(uint8_t value);
        static void writeM7A(uint8_t value);
        static void writeM7B(uint8_t value);
        static void writeM7C(uint8_t value);
        static void writeM7D(uint8_t value);
        static void writeM7X(uint8_t value);
        static void writeM7Y(uint8_t value);
        static void writeCGADD(uint8_t value);
        static void writeCGDATA(uint8_t value);
        static void writeW12SEL(uint8_t value);
        static void writeW34SEL(uint8_t value);
        static void writeWOBJSEL(uint8_t value);
        static void writeWH0(uint8_t value);
        static void writeWH1(uint8_t value);
        static void writeWH2(uint8_t value);
        static void writeWH3(uint8_t value);
        static void writeWBGLOG(uint8_t value);
        static void writeWOBJLOG(uint8_t value);
        static void writeTM(uint8_t value);
        static void writeTS(uint8_t value);
        static void writeTMW(uint8_t value);
        static void writeTSW(uint8_t value);
        static void writeCGWSEL(uint8_t value);
        static void writeCGADSUB(uint8_t value);
        static void writeCOLDATA(uint8_t value);
        static void writeSETINI(uint8_t value);

        // Data Read Functions
        static uint8_t readMPYL();
        static uint8_t readMPYM();
        static uint8_t readMPYH();
        static uint8_t readSLHV();
        static uint8_t readOAMDATAREAD();
        static uint8_t readVMDATAREAD();
        static uint8_t readCGDATAREAD();
        static uint8_t readOPHCT();
        static uint8_t readOPVCT();
        static uint8_t readSTAT77();
        static uint8_t readSTAT78();

        // APU IO Ports
        static void writeAPUIO0(uint8_t value);
        static void writeAPUIO1(uint8_t value);
        static void writeAPUIO2(uint8_t value);
        static void writeAPUIO3(uint8_t value);

        // DMA/HDMA Registers
        static void writeWMDATA(uint8_t value);
        static void writeWMADDL(uint8_t value);
        static void writeWMADDH(uint8_t value);
        static void writeWMADDX(uint8_t value);
    };

} // namespace SNES