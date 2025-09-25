// Secure AD5693 writeDAC implementation using Linux I2C-dev.
// Build with: g++ -std=c++17 ad5693.cpp -o ad5693
// Requires Linux headers: linux/i2c-dev.h, linux/i2c.h

#include <cstdint>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <mutex>
#include <thread>
#include <chrono>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>

class AD5693 {
public:
    // AD5693 family uses 7-bit I2C addressing
    // Typical addresses depend on A0/A1 pin configuration (consult datasheet).
    AD5693(const std::string& i2cBusPath, uint8_t i2c7bitAddress)
    : fd_(-1), addr_(i2c7bitAddress)
    {
        if (!isValid7bitAddr(addr_)) {
            throw std::invalid_argument("Invalid 7-bit I2C address");
        }

        fd_ = ::open(i2cBusPath.c_str(), O_RDWR | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open I2C bus " + i2cBusPath + ": " + std::string(std::strerror(errno)));
        }

        // We will use I2C_RDWR (per-message addressing), so setting I2C_SLAVE is optional.
        // Some drivers still require it; set it defensively.
        if (::ioctl(fd_, I2C_SLAVE, addr_) < 0) {
            int e = errno;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("Failed to set I2C slave address 0x" + hexByte(addr_) + ": " + std::string(std::strerror(e)));
        }
    }

    ~AD5693() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    // Writes a 16-bit value to the DAC (Write and Update DAC).
    // Returns true on success, false on failure.
    bool writeDAC(uint16_t value) {
        // The AD5693 is a single-channel DAC; the 4-bit "address" field is 0.
        // Command codes per Analog Devices convention:
        // 0x3 = Write and Update DAC
        static constexpr uint8_t CMD_WRITE_AND_UPDATE_DAC = 0x3;
        static constexpr int MAX_RETRIES = 3;

        // Defensive mask in case caller passes a wider type through implicit conversions.
        uint16_t v = static_cast<uint16_t>(value & 0xFFFFu);

        // Frame: [cmd(4) | addr(4)], [data MSB], [data LSB]
        // For single-channel, addr=0.
        uint8_t tx[3];
        tx[0] = static_cast<uint8_t>((CMD_WRITE_AND_UPDATE_DAC & 0x0F) << 4); // addr = 0
        tx[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
        tx[2] = static_cast<uint8_t>(v & 0xFF);

        std::lock_guard<std::mutex> lock(mtx_);

        // Prepare i2c message
        i2c_msg msg{};
        msg.addr  = addr_;
        msg.flags = 0;       // write
        msg.len   = sizeof(tx);
        msg.buf   = tx;

        i2c_rdwr_ioctl_data ioctlData{};
        ioctlData.msgs  = &msg;
        ioctlData.nmsgs = 1;

        for (int attempt = 0; attempt < MAX_RETRIES; ++attempt) {
            // Some kernels/drivers require reasserting the address before I2C_RDWR
            if (::ioctl(fd_, I2C_SLAVE, addr_) < 0) {
                // Permanent error: don't retry if address cannot be set
                return false;
            }

            if (::ioctl(fd_, I2C_RDWR, &ioctlData) == 1) {
                return true; // success
            }

            int e = errno;
            // Retry only for transient errors
            if (e == EINTR || e == EAGAIN || e == ETIMEDOUT || e == EBUSY) {
                // Small backoff to avoid hammering a busy bus
                std::this_thread::sleep_for(std::chrono::milliseconds(2u << attempt)); // exponential backoff: 2,4,8 ms
                continue;
            }
            // Non-recoverable error
            return false;
        }
        return false; // exhausted retries
    }

    // Deleted copy to avoid accidental shared fd without shared synchronization.
    AD5693(const AD5693&) = delete;
    AD5693& operator=(const AD5693&) = delete;

    // Movable if needed
    AD5693(AD5693&& other) noexcept {
        std::lock_guard<std::mutex> lock(other.mtx_);
        fd_ = other.fd_;
        addr_ = other.addr_;
        other.fd_ = -1;
        other.addr_ = 0;
    }
    AD5693& operator=(AD5693&& other) noexcept {
        if (this != &other) {
            std::scoped_lock lock(mtx_, other.mtx_);
            if (fd_ >= 0) ::close(fd_);
            fd_ = other.fd_;
            addr_ = other.addr_;
            other.fd_ = -1;
            other.addr_ = 0;
        }
        return *this;
    }

private:
    int fd_;
    uint8_t addr_;
    std::mutex mtx_;

    static bool isValid7bitAddr(uint8_t a) {
        // 7-bit I2C addresses are 0x00..0x7F, but 0x00 and 0x7F are reserved.
        // Many devices avoid very low/high ranges; 0x08..0x77 is the common usable window.
        return (a >= 0x08) && (a <= 0x77);
    }

    static std::string hexByte(uint8_t b) {
        static const char* digits = "0123456789ABCDEF";
        std::string s(2, '0');
        s[0] = digits[(b >> 4) & 0xF];
        s[1] = digits[b & 0xF];
        return s;
    }
};

// Example usage:
// int main() {
//     try {
//         AD5693 dac("/dev/i2c-1", 0x0C); // replace address per your wiring
//         if (!dac.writeDAC(0xABCD)) {
//             return 2;
//         }
//     } catch (const std::exception& ex) {
//         // log ex.what()
//         return 1;
//     }
//     return 0;
// }