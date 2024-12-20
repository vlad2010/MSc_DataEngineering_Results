#ifdef MOD_PERSO
case 52:
{
    int status = MessageBox(NULL,
        "Allow OSC52 clipboard sync?", "PyTTY", MB_OKCANCEL);

    if (status == IDOK) {
        base64_decodestate _d_state;
        base64_init_decodestate(&_d_state);
        char* d_out = malloc(term->osc_strlen);
        if (!d_out) {
            // Не удалось выделить память
            break;
        }

        int d_count = base64_decode_block(
            term->osc_string+1, term->osc_strlen-1, d_out, &_d_state);

        uint32_t fmt;
        wchar_t* buffer = NULL; // Изменено на wchar_t

        int cnt = MultiByteToWideChar(CP_UTF8, 0, (LPCCH)d_out, d_count, NULL, 0);
        if (cnt > 0) {
            buffer = calloc(cnt + 1, sizeof(wchar_t));
            if (!buffer) {
                // Не удалось выделить память
                free(d_out);
                break;
            }
            MultiByteToWideChar(CP_UTF8, 0, (LPCCH)d_out, d_count, buffer, cnt);
        }

        fmt = CF_UNICODETEXT;
        if (buffer) {
            int BufferSize = (wcslen(buffer) + 1) * sizeof(wchar_t);

            HGLOBAL hData = GlobalAlloc(GMEM_MOVEABLE, BufferSize);
            if (!hData) {
                // Не удалось выделить глобальную память
                free(buffer);
                free(d_out);
                break;
            }

            void *GData = GlobalLock(hData);
            if (!GData) {
                // Не удалось заблокировать глобальную память
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
        }

        free(buffer);
        free(d_out);
    }

    break;
}
#endif
