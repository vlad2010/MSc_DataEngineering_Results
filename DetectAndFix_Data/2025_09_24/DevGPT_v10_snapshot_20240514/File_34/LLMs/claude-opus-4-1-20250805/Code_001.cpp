#include <iostream>
#include <windows.h>

HHOOK hHook;

LRESULT CALLBACK HookCallback(int nCode, WPARAM wParam, LPARAM lParam)
{
    if (nCode == HC_ACTION && wParam == HSHELL_WINDOWCREATED)
    {
        HWND hWnd = reinterpret_cast<HWND>(lParam);
        char processName[MAX_PATH];
        GetWindowTextA(hWnd, processName, MAX_PATH);
        std::cout << "Process created: " << processName << std::endl;
    }

    return CallNextHookEx(hHook, nCode, wParam, lParam);
}

int main()
{
    // Set the hook
    hHook = SetWindowsHookEx(WH_SHELL, HookCallback, NULL, 0);

    if (hHook == NULL)
    {
        std::cout << "Failed to set hook." << std::endl;
        return 1;
    }

    // Message loop
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // Unhook the hook
    UnhookWindowsHookEx(hHook);

    return 0;
}