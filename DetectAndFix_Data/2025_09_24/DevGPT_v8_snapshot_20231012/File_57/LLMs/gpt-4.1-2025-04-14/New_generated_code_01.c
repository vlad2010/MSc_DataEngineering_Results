#ifdef MOD_PERSO
case 52:
{
    int status = MessageBox(NULL,
        "Allow OSC52 clipboard sync?", "PyTTY", MB_OKCANCEL);

    if (status == IDOK) {
        base64_decodestate _d_state;
        base64_init_decodestate(&_d_state);

        // Allocate enough space for decoded data + 1 for null-terminator
        char* d_out = malloc(term->osc_strlen + 1);
        if (!d_out) {
            // Failed to allocate memory
            break;
        }

        int d_count = base64_decode_block(
            term->osc_string+1, term->osc_strlen-1, d_out, &_d_state);

        // Ensure null-termination for safety
        d_out[d_count] = '\0';

        // Validate UTF-8 (simple check, can be replaced with robust validation)
        // Here, we just check that MultiByteToWideChar will not fail due to invalid UTF-8
        int cnt = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, (LPCCH)d_out, d_count, NULL, 0);
        if (cnt <= 0) {
            // Invalid UTF-8 or conversion error
            free(d_out);
            break;
        }

        wchar_t* buffer = calloc(cnt + 1, sizeof(wchar_t));
        if (!buffer) {
            // Failed to allocate memory
            free(d_out);
            break;
        }

        int written = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, (LPCCH)d_out, d_count, buffer, cnt);
        if (written != cnt) {
            // Conversion error
            free(buffer);
            free(d_out);
            break;
        }
        buffer[cnt] = L'\0'; // Null-terminate

        uint32_t fmt = CF_UNICODETEXT;

        int BufferSize = (wcslen(buffer) + 1) * sizeof(wchar_t);

        HGLOBAL hData = GlobalAlloc(GMEM_MOVEABLE, BufferSize);
        if (!hData) {
            // Failed to allocate global memory
            free(buffer);
            free(d_out);
            break;
        }

        void *GData = GlobalLock(hData);
        if (!GData) {
            // Failed to lock global memory
            GlobalFree(hData);
            free(buffer);
            free(d_out);
            break;
        }

        memcpy(GData, buffer, BufferSize);
        GlobalUnlock(hData);

        if (OpenClipboard(NULL)) {
            EmptyClipboard();
            if (!SetClipboardData(fmt, (HANDLE)hData)) {
                // Clipboard did not take ownership, free memory
                GlobalFree(hData);
            }
            CloseClipboard();
        } else {
            // Could not open clipboard, free memory
            GlobalFree(hData);
        }

        free(buffer);
        free(d_out);
    }

    break;
}
#endif