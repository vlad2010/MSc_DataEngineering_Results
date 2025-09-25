#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

#ifdef _WIN32
  #define NOMINMAX
  #include <windows.h> // SecureZeroMemory
#endif

#ifdef __STDC_LIB_EXT1__
  #include <string.h>  // memset_s
#endif

#include <curl/curl.h>

// Best-effort secure memory wipe
static inline void secure_memzero(void* p, size_t n) {
    if (p == nullptr || n == 0) return;
#if defined(__STDC_LIB_EXT1__)
    memset_s(p, n, 0, n);
#elif defined(_WIN32)
    SecureZeroMemory(p, n);
#else
    volatile unsigned char* vp = static_cast<volatile unsigned char*>(p);
    while (n--) { *vp++ = 0; }
#endif
}

// Safe multiply for size_t with overflow handling.
// Returns true if overflow occurred, false otherwise. Result stored in out.
static inline bool mul_overflow_size_t(size_t a, size_t b, size_t& out) {
#if defined(__has_builtin)
  #if __has_builtin(__builtin_mul_overflow)
    return __builtin_mul_overflow(a, b, &out);
  #else
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) return true;
    out = a * b; return false;
  #endif
#else
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) return true;
    out = a * b; return false;
#endif
}

// Libcurl-compatible read callback.
// ptr: destination buffer provided by libcurl
// size * nmemb: total available space in ptr (bytes)
// userdata: expected to be std::string* holding the remaining body to send
size_t read_data_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    if (ptr == nullptr || userdata == nullptr) {
        return CURL_READFUNC_ABORT; // defensive: abort on invalid inputs
    }

    auto* data = static_cast<std::string*>(userdata);
    if (data->empty()) {
        return 0; // EOF
    }

    // Compute available buffer size safely
    size_t bufsize = 0;
    if (mul_overflow_size_t(size, nmemb, bufsize)) {
        // In the extremely unlikely case of overflow, clamp to max representable.
        bufsize = std::numeric_limits<size_t>::max();
    }

    // Number of bytes we will copy this call
    const size_t len = std::min(bufsize, data->size());
    if (len == 0) {
        return 0; // nothing to send now (treat as EOF for libcurl semantics)
    }

    // Copy from the string into libcurl's buffer
    // Note: &(*data)[0] is safe when len > 0 (string is non-empty)
    char* src = &(*data)[0];
    std::memcpy(ptr, src, len);

    // Best-effort scrub the portion we just copied before removing it
    secure_memzero(src, len);
    data->erase(0, len);

    // Avoid logging sensitive content; if absolutely necessary, log length only and behind a guarded build flag.
    // #ifdef ENABLE_SAFE_DEBUG
    // std::cout << "[Sent Body bytes]: " << len << std::endl;
    // #endif

    return len;
}