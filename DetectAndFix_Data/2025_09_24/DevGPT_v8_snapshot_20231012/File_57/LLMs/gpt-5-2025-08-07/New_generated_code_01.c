#include <windows.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Assume these are provided by your base64 library.
typedef struct base64_decodestate base64_decodestate;
void base64_init_decodestate(base64_decodestate* state);
int base64_decode_block(const char* code_in, const int length_in, char* plaintext_out, base64_decodestate* state);

// Policy limit to prevent DoS from massive clipboard payloads.
// Tune as appropriate for your application.
#define MAX_OSC52_DECODED_BYTES (8u * 1024u * 1024u)  // 8 MiB

#ifdef MOD_PERSO
case 52:
{
    // Ask user for permission (use wide for better i18n and consistency with Unicode).
    int status = MessageBoxW(NULL,
        L"Allow OSC52 clipboard sync?",
        L"PyTTY",
        MB_OKCANCEL | MB_ICONQUESTION | MB_SETFOREGROUND);

    if (status != IDOK) {
        break;
    }

    // Validate input
    if (!term || !term->osc_string || term->osc_strlen <= 1) {
        // Nothing usable or clearly invalid (we subtract 1 below).
        break;
    }

    // Compute sizes in size_t to avoid signed overflow/underflow.
    const size_t in_total_len = (size_t)term->osc_strlen;
    const size_t skip = 1; // Existing behavior: skip first byte of osc_string.
    const size_t enc_len = in_total_len - skip;

    // Compute a safe upper bound on decoded size: ((enc_len + 3) / 4) * 3
    // and check for overflow and policy cap.
    size_t dec_cap = 0;
    if (enc_len > SIZE_MAX - 3) {
        // Prevent enc_len + 3 overflow
        break;
    }
    dec_cap = ((enc_len + 3) / 4) * 3;

    if (dec_cap == 0 || dec_cap > MAX_OSC52_DECODED_BYTES) {
        // Reject empty or excessively large payloads
        break;
    }

    char* d_out = (char*)malloc(dec_cap);
    if (!d_out) {
        break;
    }

    // Decode base64
    base64_decodestate dstate;
    base64_init_decodestate(&dstate);
    int d_count = base64_decode_block(term->osc_string + skip, (int)enc_len, d_out, &dstate);
    if (d_count <= 0 || (size_t)d_count > dec_cap) {
        // Decoding failed or returned impossible length
        SecureZeroMemory(d_out, dec_cap);
        free(d_out);
        break;
    }

    // Validate and convert UTF-8 -> UTF-16. Reject invalid UTF-8.
    // MultiByteToWideChar returns the number of WCHARs required (excluding terminator).
    int wide_len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, d_out, d_count, NULL, 0);
    if (wide_len <= 0) {
        SecureZeroMemory(d_out, d_count);
        free(d_out);
        break;
    }

    // Check for overflow in allocation: (wide_len + 1) * sizeof(wchar_t)
    if ((size_t)wide_len > (SIZE_MAX / sizeof(wchar_t)) - 1) {
        SecureZeroMemory(d_out, d_count);
        free(d_out);
        break;
    }

    wchar_t* buffer = (wchar_t*)malloc(((size_t)wide_len + 1) * sizeof(wchar_t));
    if (!buffer) {
        SecureZeroMemory(d_out, d_count);
        free(d_out);
        break;
    }

    int written = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, d_out, d_count, buffer, wide_len);
    if (written != wide_len) {
        SecureZeroMemory(buffer, ((size_t)wide_len + 1) * sizeof(wchar_t));
        free(buffer);
        SecureZeroMemory(d_out, d_count);
        free(d_out);
        break;
    }
    buffer[wide_len] = L'\0'; // Explicitly NUL-terminate.

    // Prepare clipboard data as CF_UNICODETEXT
    UINT fmt = CF_UNICODETEXT;

    // CF_UNICODETEXT expects a terminating NUL; compute bytes including NUL.
    size_t bytes_to_copy = ((size_t)wide_len + 1) * sizeof(wchar_t);

    HGLOBAL hData = GlobalAlloc(GMEM_MOVEABLE, bytes_to_copy);
    if (!hData) {
        SecureZeroMemory(buffer, bytes_to_copy);
        free(buffer);
        SecureZeroMemory(d_out, d_count);
        free(d_out);
        break;
    }

    void* gptr = GlobalLock(hData);
    if (!gptr) {
        GlobalFree(hData);
        SecureZeroMemory(buffer, bytes_to_copy);
        free(buffer);
        SecureZeroMemory(d_out, d_count);
        free(d_out);
        break;
    }

    memcpy(gptr, buffer, bytes_to_copy);
    GlobalUnlock(hData);

    if (OpenClipboard(NULL)) {
        EmptyClipboard();
        if (!SetClipboardData(fmt, hData)) {
            // Only free if SetClipboardData failed; otherwise ownership is transferred to the clipboard.
            GlobalFree(hData);
        }
        CloseClipboard();
    } else {
        GlobalFree(hData);
    }

    // Cleanup
    SecureZeroMemory(buffer, bytes_to_copy);
    free(buffer);
    SecureZeroMemory(d_out, d_count);
    free(d_out);

    break;
}
#endif