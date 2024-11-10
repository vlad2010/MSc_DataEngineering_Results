#ifdef MOD_PERSO
case 52:
{
    int status = MessageBox(NULL,
        "Allow OSC52 clipboard sync?", "PyTTY", MB_OKCANCEL);

    if (status == IDOK) {
        if (term->osc_strlen <= 0 || !term->osc_string) {
            // Invalid data
            break;
        }

        base64_decodestate _d_state;
        base64_init_decodestate(&_d_state);
        char* d_out = malloc(term->osc_strlen);
        if (!d_out) {
            // Failed to allocate memory
            break;
        }

        int d_count = base64_decode_block(
            term->osc_string+1, term->osc_strlen-1, d_out, &_d_state);

        // Ensure d_count is within expected bounds:
        if (d_count <= 0) {
            free(d_out);
            break;
        }

        wchar_t* buffer = NULL;

        int cnt = MultiByteToWideChar(CP_UTF8, 0, (LPCCH)d_out, d_count, NULL, 0);
        if (cnt <= 0) {
            free(d_out);
            break;
        }

        buffer = calloc(cnt + 1, sizeof(wchar_t));
        if (!buffer) {
            free(d_out);
            break;
        }

        MultiByteToWideChar(CP_UTF8, 0, (LPCCH)d_out, d_count, buffer, cnt);

        uint32_t fmt = CF_UNICODETEXT;

        int BufferSize = (wcslen(buffer) + 1) * sizeof(wchar_t);

        if (BufferSize <= 0) {
            free(buffer);
            free(d_out);
            break;
        }

        HGLOBAL hData = GlobalAlloc(GMEM_MOVEABLE, BufferSize);
        if (!hData) {
            free(buffer);
            free(d_out);
            break;
        }

        void *GData = GlobalLock(hData);
        if (!GData) {
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
                GlobalFree(hData);
            }
            CloseClipboard();
        } else {
            GlobalFree(hData);
        }

        free(buffer);
        free(d_out);
    }
    break;
}
#endif