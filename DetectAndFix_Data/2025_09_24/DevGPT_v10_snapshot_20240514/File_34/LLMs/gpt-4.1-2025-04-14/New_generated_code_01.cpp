#include <iostream>
#include <windows.h>
#include <psapi.h> // For GetModuleFileNameEx

#pragma comment(lib, "Psapi.lib")

HHOOK hHook = NULL;

LRESULT CALLBACK HookCallback(int nCode, WPARAM wParam, LPARAM lParam)
{
    if (nCode == HC_ACTION && wParam == HSHELL_WINDOWCREATED)
    {
        HWND hWnd = reinterpret_cast<HWND>(lParam);

        // Get process ID from window handle
        DWORD processId = 0;
        GetWindowThreadProcessId(hWnd, &processId);
        if (processId == 0)
        {
            std::cerr << "Failed to get process ID." << std::endl;
            return CallNextHookEx(hHook, nCode, wParam, lParam);
        }

        // Open the process with minimal rights
        HANDLE hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, FALSE, processId);
        if (hProcess == NULL)
        {
            std::cerr << "Failed to open process. Error: " << GetLastError() << std::endl;
            return CallNextHookEx(hHook, nCode, wParam, lParam);
        }

        // Get the process executable name
        char processName[MAX_PATH] = "<unknown>";
        DWORD size = sizeof(processName);
        if (GetModuleFileNameExA(hProcess, NULL, processName, size) == 0)
        {
            std::cerr << "Failed to get process name. Error: " << GetLastError() << std::endl;
            CloseHandle(hProcess);
            return CallNextHookEx(hHook, nCode, wParam, lParam);
        }

        std::cout << "Process created: " << processName << std::endl;

        // Close the process handle to avoid resource leak
        CloseHandle(hProcess);
    }

    return CallNextHookEx(hHook, nCode, wParam, lParam);
}

int main()
{
    // Set the hook
    hHook = SetWindowsHookEx(WH_SHELL, HookCallback, NULL, 0);

    if (hHook == NULL)
    {
        std::cerr << "Failed to set hook. Error: " << GetLastError() << std::endl;
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
    if (hHook != NULL)
    {
        UnhookWindowsHookEx(hHook);
        hHook = NULL;
    }

    return 0;
}