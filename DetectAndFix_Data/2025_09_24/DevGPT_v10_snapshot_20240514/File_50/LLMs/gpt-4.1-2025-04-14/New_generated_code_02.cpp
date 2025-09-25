// snes_ppu.h
#pragma once

#include <cstdint>
#include <array>
#include <algorithm>

namespace SNES {

    // PPU Register Addresses
    constexpr uint16_t INIDISP      = 0x2100;
    constexpr uint16_t OBSEL        = 0x2101;
    constexpr uint16_t OAMADDL      = 0x2102;
    constexpr uint16_t OAMADDH      = 0x2103;
    constexpr uint16_t OAMDATA      = 0x2104;
    constexpr uint16_t BGMODE       = 0x2105;
    constexpr uint16_t MOSAIC       = 0x2106;
    constexpr uint16_t BG1SC        = 0x2107;
    constexpr uint16_t BG2SC        = 0x2108;
    constexpr uint16_t BG3SC        = 0x2109;
    constexpr uint16_t BG4SC        = 0x210A;
    constexpr uint16_t BG12NBA      = 0x210B;
    constexpr uint16_t BG34NBA      = 0x210C;
    constexpr uint16_t BG1HOFS      = 0x210D;
    constexpr uint16_t BG1VOFS      = 0x210E;
    constexpr uint16_t BG2HOFS      = 0x210F;
    constexpr uint16_t BG2VOFS      = 0x2110;
    constexpr uint16_t BG3HOFS      = 0x2111;
    constexpr uint16_t BG3VOFS      = 0x2112;
    constexpr uint16_t BG4HOFS      = 0x2113;
    constexpr uint16_t BG4VOFS      = 0x2114;
    constexpr uint16_t VMAIN        = 0x2115;
    constexpr uint16_t VMADDL       = 0x2116;
    constexpr uint16_t VMADDH       = 0x2117;
    constexpr uint16_t VMDATA       = 0x2118;
    constexpr uint16_t M7HOFS       = 0x211B;
    constexpr uint16_t M7VOFS       = 0x211C;
    constexpr uint16_t M7A          = 0x211D;
    constexpr uint16_t M7B          = 0x211E;
    constexpr uint16_t M7C          = 0x211F;
    constexpr uint16_t M7D          = 0x2120;
    // Note: 0x2121 and 0x2122 are used for both M7X/M7Y and CGADD/CGDATA. Clarify with comments.
    constexpr uint16_t M7X          = 0x2121; // Alias for CGADD
    constexpr uint16_t M7Y          = 0x2122; // Alias for CGDATA
    constexpr uint16_t CGADD        = 0x2121;
    constexpr uint16_t CGDATA       = 0x2122;
    constexpr uint16_t W12SEL       = 0x2123;
    constexpr uint16_t W34SEL       = 0x2124;
    constexpr uint16_t WOBJSEL      = 0x2125;
    constexpr uint16_t WH0          = 0x2126;
    constexpr uint16_t WH1          = 0x2127;
    constexpr uint16_t WH2          = 0x2128;
    constexpr uint16_t WH3          = 0x2129;
    constexpr uint16_t WBGLOG       = 0x212A;
    constexpr uint16_t WOBJLOG      = 0x212B;
    constexpr uint16_t TM           = 0x212C;
    constexpr uint16_t TS           = 0x212D;
    constexpr uint16_t TMW          = 0x212E;
    constexpr uint16_t TSW          = 0x212F;
    constexpr uint16_t CGWSEL       = 0x2130;
    constexpr uint16_t CGADSUB      = 0x2131;
    constexpr uint16_t COLDATA      = 0x2132;
    constexpr uint16_t SETINI       = 0x2133;
    constexpr uint16_t MPYL         = 0x2134;
    constexpr uint16_t MPYM         = 0x2135;
    constexpr uint16_t MPYH         = 0x2136;
    constexpr uint16_t SLHV         = 0x2137;
    constexpr uint16_t OAMDATAREAD  = 0x2138;
    constexpr uint16_t VMDATAREAD   = 0x2139;
    constexpr uint16_t CGDATAREAD   = 0x213A;
    constexpr uint16_t OPHCT        = 0x213B;
    constexpr uint16_t OPVCT        = 0x213C;
    constexpr uint16_t STAT77       = 0x213D;
    constexpr uint16_t STAT78       = 0x213E;
    constexpr uint16_t APUIO0       = 0x2140;
    constexpr uint16_t APUIO1       = 0x2141;
    constexpr uint16_t APUIO2       = 0x2142;
    constexpr uint16_t APUIO3       = 0x2143;
    constexpr uint16_t WMDATA       = 0x2180;
    constexpr uint16_t WMADDL       = 0x2181;
    constexpr uint16_t WMADDH       = 0x2182;
    constexpr uint16_t WMADDX       = 0x2183;

    // List of valid PPU register addresses for validation
    constexpr std::array<uint16_t, 52> VALID_PPU_ADDRESSES = {
        INIDISP, OBSEL, OAMADDL, OAMADDH, OAMDATA, BGMODE, MOSAIC, BG1SC, BG2SC, BG3SC, BG4SC,
        BG12NBA, BG34NBA, BG1HOFS, BG1VOFS, BG2HOFS, BG2VOFS, BG3HOFS, BG3VOFS, BG4HOFS, BG4VOFS,
        VMAIN, VMADDL, VMADDH, VMDATA, M7HOFS, M7VOFS, M7A, M7B, M7C, M7D, CGADD, CGDATA,
        W12SEL, W34SEL, WOBJSEL, WH0, WH1, WH2, WH3, WBGLOG, WOBJLOG, TM, TS, TMW, TSW,
        CGWSEL, CGADSUB, COLDATA, SETINI, MPYL, MPYM, MPYH, SLHV, OAMDATAREAD, VMDATAREAD,
        CGDATAREAD, OPHCT, OPVCT, STAT77, STAT78, APUIO0, APUIO1, APUIO2, APUIO3,
        WMDATA, WMADDL, WMADDH, WMADDX
    };

    // Helper function to check if address is valid
    constexpr bool is_valid_ppu_address(uint16_t address) {
        // Use binary search for efficiency (array must be sorted)
        // For constexpr, use a simple loop
        for (auto valid : VALID_PPU_ADDRESSES) {
            if (address == valid) return true;
        }
        return false;
    }

    // SNES PPU Class
    class PPU {
    public:
        static void write(uint16_t address, uint8_t value) {
            if (!is_valid_ppu_address(address)) {
                // Optionally: log error, throw exception, or handle as needed
                // For safety, do nothing or abort
                // throw std::invalid_argument("Invalid PPU register address");
                return;
            }
            *reinterpret_cast<volatile uint8_t*>(address) = value;
        }

        static uint8_t read(uint16_t address) {
            if (!is_valid_ppu_address(address)) {
                // Optionally: log error, throw exception, or handle as needed
                // throw std::invalid_argument("Invalid PPU register address");
                return 0;
            }
            return *reinterpret_cast<volatile uint8_t*>(address);
        }

        static void setINIDISP(uint8_t brightness, bool displayEnabled) {
            write(INIDISP, (brightness & 0x0F) | (displayEnabled ? 0x80 : 0x00));
        }

        static void setBGMODE(uint8_t mode) {
            write(BGMODE, mode & 0x07);
        }

        static void setOBSEL(uint8_t size, uint8_t address) {
            write(OBSEL, (size << 5) | (address & 0x1F));
        }

        // Add other register manipulation functions as needed
    };

} // namespace SNES