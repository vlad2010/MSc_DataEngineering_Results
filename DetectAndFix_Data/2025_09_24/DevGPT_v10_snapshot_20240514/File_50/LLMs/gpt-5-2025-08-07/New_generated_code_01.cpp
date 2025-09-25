// snes_ppu.h (hardened)
// Security fixes:
// - No integer-to-pointer dereference (CWE-587/CWE-704)
// - Validate addresses against allow-list (CWE-787/788)
// - Correct Mode 7 register addresses (CWE-682/CWE-840)
// - Mask bitfields in setters (CWE-681/682)
// - Optional thread-safety (CWE-362)

#pragma once

#include <cstdint>
#include <array>
#include <mutex>
#include <stdexcept>
#include <functional>

namespace SNES {

    // PPU Register Addresses (corrected)
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

    // Correct Mode 7 registers
    constexpr uint16_t M7A          = 0x211B;
    constexpr uint16_t M7B          = 0x211C;
    constexpr uint16_t M7C          = 0x211D;
    constexpr uint16_t M7D          = 0x211E;
    constexpr uint16_t M7X          = 0x211F;
    constexpr uint16_t M7Y          = 0x2120;

    // Note: Do not define M7HOFS/M7VOFS; BG1HOFS/BG1VOFS are used for Mode 7 scrolling.

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

    // Helper: Validate PPU register address ranges.
    inline bool isValidPPUAddress(uint16_t address) {
        // SNES PPU/APU I/O windows
        return (address >= 0x2100 && address <= 0x213F) ||
               (address >= 0x2140 && address <= 0x2143) ||
               (address >= 0x2180 && address <= 0x2183);
    }

    // SNES PPU Class with safe backend
    class PPU {
    public:
        // Backend interface: user can inject real MMIO handlers if needed.
        // Defaults to a safe internal emulation store.
        using WriteFn = std::function<void(uint16_t /*address*/, uint8_t /*value*/)>;
        using ReadFn  = std::function<uint8_t(uint16_t /*address*/)>;

        // Install custom backend (e.g., real MMIO). Caller is responsible for safety.
        static void setBackend(WriteFn writer, ReadFn reader) {
            std::lock_guard<std::mutex> lock(mutex());
            s_write = std::move(writer);
            s_read  = std::move(reader);
        }

        static void write(uint16_t address, uint8_t value) {
            if (!isValidPPUAddress(address)) {
                throw std::out_of_range("PPU::write: invalid PPU register address");
            }
            std::lock_guard<std::mutex> lock(mutex());
            s_write(address, value);
        }

        static uint8_t read(uint16_t address) {
            if (!isValidPPUAddress(address)) {
                throw std::out_of_range("PPU::read: invalid PPU register address");
            }
            std::lock_guard<std::mutex> lock(mutex());
            return s_read(address);
        }

        static void setINIDISP(uint8_t brightness, bool displayEnabled) {
            // brightness is 4 bits (0-15), displayEnabled maps to bit 7
            const uint8_t b = static_cast<uint8_t>(brightness & 0x0F);
            const uint8_t en = displayEnabled ? 0x80 : 0x00;
            write(INIDISP, static_cast<uint8_t>(b | en));
        }

        static void setBGMODE(uint8_t mode) {
            // Mode is 3 bits (0-7)
            write(BGMODE, static_cast<uint8_t>(mode & 0x07));
        }

        static void setOBSEL(uint8_t size, uint8_t nameBaseAddress) {
            // size: 3 bits at [7:5], name base: 5 bits at [4:0]
            const uint8_t sz  = static_cast<uint8_t>((size & 0x07) << 5);
            const uint8_t nba = static_cast<uint8_t>(nameBaseAddress & 0x1F);
            write(OBSEL, static_cast<uint8_t>(sz | nba));
        }

        // Additional register manipulation helpers can be added with proper masking and validation.

    private:
        // Default safe backend: internal emulated register file (no raw pointer derefs).
        static WriteFn& defaultWriter() {
            static WriteFn fn = [](uint16_t addr, uint8_t val) {
                s_regs[addr] = val;
            };
            return fn;
        }
        static ReadFn& defaultReader() {
            static ReadFn fn = [](uint16_t addr) -> uint8_t {
                return s_regs[addr];
            };
            return fn;
        }

        static std::mutex& mutex() {
            static std::mutex m;
            return m;
        }

        static WriteFn& writerRef() {
            static WriteFn w = defaultWriter();
            return w;
        }

        static ReadFn& readerRef() {
            static ReadFn r = defaultReader();
            return r;
        }

        static WriteFn& s_write;
        static ReadFn&  s_read;

        // Emulated register space: index by physical address after allow-list check.
        // Size: up to the highest address we allow (0x2183). Allocate 0x2184 bytes.
        static std::array<uint8_t, 0x2184>& regsRef() {
            static std::array<uint8_t, 0x2184> arr{}; // zero-initialized
            return arr;
        }
        static std::array<uint8_t, 0x2184>& s_regs;
    };

    // Static member definitions
    inline PPU::WriteFn& PPU::s_write = PPU::writerRef();
    inline PPU::ReadFn&  PPU::s_read  = PPU::readerRef();
    inline std::array<uint8_t, 0x2184>& PPU::s_regs = PPU::regsRef();

} // namespace SNES