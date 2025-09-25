// snes_ppu.h
#pragma once

#include <cstdint>
#include <array>
#include <stdexcept>
#include <cstring>

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
    constexpr uint16_t M7X          = 0x2121;
    constexpr uint16_t M7Y          = 0x2122;
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

    // SNES PPU Class with proper memory management
    class PPU {
    private:
        // Define the PPU register memory range
        static constexpr uint16_t PPU_REG_START = 0x2100;
        static constexpr uint16_t PPU_REG_END = 0x2183;
        static constexpr size_t PPU_REG_SIZE = PPU_REG_END - PPU_REG_START + 1;
        
        // Simulated PPU register memory
        std::array<uint8_t, PPU_REG_SIZE> registers;
        
        // Singleton instance for static access
        static PPU& getInstance() {
            static PPU instance;
            return instance;
        }
        
        // Private constructor for singleton
        PPU() {
            // Initialize all registers to 0
            registers.fill(0);
        }
        
        // Validate and convert address to array index
        static size_t validateAndConvertAddress(uint16_t address) {
            if (address < PPU_REG_START || address > PPU_REG_END) {
                throw std::out_of_range("PPU register address out of range: 0x" + 
                                       std::to_string(address));
            }
            return address - PPU_REG_START;
        }
        
    public:
        // Delete copy constructor and assignment operator for singleton
        PPU(const PPU&) = delete;
        PPU& operator=(const PPU&) = delete;
        
        static void write(uint16_t address, uint8_t value) {
            try {
                size_t index = validateAndConvertAddress(address);
                PPU& ppu = getInstance();
                
                // Add write protection for read-only registers
                if (isReadOnlyRegister(address)) {
                    // Log warning or handle as appropriate for your emulator
                    return;
                }
                
                ppu.registers[index] = value;
            } catch (const std::exception& e) {
                // Handle error appropriately (log, ignore, etc.)
                // For emulator, might want to log and continue
            }
        }

        static uint8_t read(uint16_t address) {
            try {
                size_t index = validateAndConvertAddress(address);
                PPU& ppu = getInstance();
                
                // Handle write-only registers
                if (isWriteOnlyRegister(address)) {
                    // Return open bus or last value depending on hardware behavior
                    return 0xFF; // Open bus value
                }
                
                return ppu.registers[index];
            } catch (const std::exception& e) {
                // Handle error appropriately
                return 0xFF; // Return open bus value on error
            }
        }
        
        static void setINIDISP(uint8_t brightness, bool displayEnabled) {
            // Validate brightness is within 4-bit range
            brightness &= 0x0F;
            write(INIDISP, brightness | (displayEnabled ? 0x00 : 0x80));
        }

        static void setBGMODE(uint8_t mode) {
            // Validate mode is within 3-bit range
            write(BGMODE, mode & 0x07);
        }

        static void setOBSEL(uint8_t size, uint8_t address) {
            // Validate parameters
            size &= 0x07;      // 3-bit size value
            address &= 0x07;   // 3-bit address value
            write(OBSEL, (size << 5) | (address << 1));
        }
        
        // Reset all registers to initial state
        static void reset() {
            PPU& ppu = getInstance();
            ppu.registers.fill(0);
            // Set default values for specific registers if needed
        }
        
    private:
        // Helper functions to identify register types
        static bool isReadOnlyRegister(uint16_t address) {
            // Read-only registers
            return (address >= MPYL && address <= MPYH) ||
                   (address >= SLHV && address <= STAT78) ||
                   (address == OAMDATAREAD) ||
                   (address == VMDATAREAD) ||
                   (address == CGDATAREAD);
        }
        
        static bool isWriteOnlyRegister(uint16_t address) {
            // Write-only registers
            return (address >= INIDISP && address <= CGDATA) ||
                   (address >= W12SEL && address <= SETINI) ||
                   (address >= WMDATA && address <= WMADDX);
        }
    };

} // namespace SNES