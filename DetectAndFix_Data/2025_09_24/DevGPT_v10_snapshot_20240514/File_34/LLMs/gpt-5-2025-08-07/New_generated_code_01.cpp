#define UNICODE
#define _UNICODE
#include <windows.h>
#include <string>
#include <iostream>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "kernel32.lib")

// Hooks installed out-of-context (no code injection into other processes)
static HWINEVENTHOOK g_hookCreate = nullptr;
static HWINEVENTHOOK g_hookShow   = nullptr;

static bool IsTopLevelWindow(HWND hwnd) {
    return hwnd && GetAncestor(hwnd, GA_ROOT) == hwnd;
}

static std::wstring GetWindowTitle(HWND hwnd) {
    // Get text length first to allocate exact size
    int len = GetWindowTextLengthW(hwnd);
    if (len <= 0) {
        // Could be 0 for empty title or cross-process control -> treat as empty
        return L"";
    }
    std::wstring title;
    title.resize(static_cast<size_t>(len) + 1, L'\0');
    int copied = GetWindowTextW(hwnd, &title[0], static_cast<int>(title.size()));
    if (copied <= 0) {
        return L"";
    }
    title.resize(static_cast<size_t>(copied));
    return title;
}

static std::wstring GetProcessImagePathFromHwnd(HWND hwnd) {
    DWORD pid = 0;
    GetWindowThreadProcessId(hwnd, &pid);
    if (pid == 0) return L"";

    HANDLE hProc = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!hProc) return L"";

    // Reasonable buffer for Windows paths; QueryFullProcessImageNameW will set actual size
    std::wstring path(32768, L'\0');
    DWORD sizeInChars = static_cast<DWORD>(path.size());
    if (QueryFullProcessImageNameW(hProc, 0, &path[0], &sizeInChars)) {
        path.resize(sizeInChars);
    } else {
        path.clear();
    }
    CloseHandle(hProc);
    return path;
}

static void LogWindowCreate(HWND hwnd) {
    if (!hwnd || !IsTopLevelWindow(hwnd) || !IsWindowVisible(hwnd)) return;

    std::wstring title = GetWindowTitle(hwnd);
    std::wstring imagePath = GetProcessImagePathFromHwnd(hwnd);

    // Basic redaction/limits to reduce potential leakage and log abuse
    const size_t kMaxLen = 4096;
    if (title.size() > kMaxLen) title = title.substr(0, kMaxLen) + L"...";
    if (imagePath.size() > kMaxLen) imagePath = imagePath.substr(0, kMaxLen) + L"...";

    std::wcout << L"Window created: "
               << (title.empty() ? L"<no title>" : title)
               << L" | Process: "
               << (imagePath.empty() ? L"<unknown>" : imagePath)
               << std::endl;
}

void CALLBACK WinEventProc(HWINEVENTHOOK, DWORD event, HWND hwnd, LONG idObject, LONG idChild,
                           DWORD /*idEventThread*/, DWORD /*dwmsEventTime*/) {
    // Only consider actual window objects, not child objects
    if (idObject != OBJID_WINDOW || idChild != 0) return;

    // React to window creation/show events
    if (event == EVENT_OBJECT_CREATE || event == EVENT_OBJECT_SHOW) {
        LogWindowCreate(hwnd);
    }
}

int wmain() {
    // Install out-of-context WinEvent hooks to avoid code injection into other processes
    g_hookCreate = SetWinEventHook(
        EVENT_OBJECT_CREATE, EVENT_OBJECT_CREATE,
        nullptr, WinEventProc, 0, 0,
        WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS
    );
    if (!g_hookCreate) {
        std::wcerr << L"Failed to set EVENT_OBJECT_CREATE hook. Error: " << GetLastError() << std::endl;
        return 1;
    }

    g_hookShow = SetWinEventHook(
        EVENT_OBJECT_SHOW, EVENT_OBJECT_SHOW,
        nullptr, WinEventProc, 0, 0,
        WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS
    );
    if (!g_hookShow) {
        std::wcerr << L"Failed to set EVENT_OBJECT_SHOW hook. Error: " << GetLastError() << std::endl;
        UnhookWinEvent(g_hookCreate);
        g_hookCreate = nullptr;
        return 1;
    }

    // Standard message loop
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    // Cleanup (also consider handling console control signals to ensure cleanup on Ctrl+C)
    if (g_hookCreate) {
        if (!UnhookWinEvent(g_hookCreate)) {
            std::wcerr << L"UnhookWinEvent(EVENT_OBJECT_CREATE) failed. Error: " << GetLastError() << std::endl;
        }
        g_hookCreate = nullptr;
    }
    if (g_hookShow) {
        if (!UnhookWinEvent(g_hookShow)) {
            std::wcerr << L"UnhookWinEvent(EVENT_OBJECT_SHOW) failed. Error: " << GetLastError() << std::endl;
        }
        g_hookShow = nullptr;
    }

    return 0;
}