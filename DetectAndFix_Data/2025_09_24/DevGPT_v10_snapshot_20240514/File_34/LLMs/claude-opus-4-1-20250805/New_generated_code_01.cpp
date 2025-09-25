#include <iostream>
#include <windows.h>
#include <string>
#include <mutex>

HHOOK hHook = NULL;
std::mutex outputMutex;

LRESULT CALLBACK HookCallback(int nCode, WPARAM wParam, LPARAM lParam)
{
    // Check if we should process this message
    if (nCode < 0) {
        return CallNextHookEx(hHook, nCode, wParam, lParam);
    }

    if (nCode == HC_ACTION && wParam == HSHELL_WINDOWCREATED)
    {
        HWND hWnd = reinterpret_cast<HWND>(lParam);
        
        // Validate window handle
        if (hWnd != NULL && IsWindow(hWnd))
        {
            // Use a larger buffer and ensure null termination
            const int BUFFER_SIZE = MAX_PATH + 1;
            char windowText[BUFFER_SIZE] = {0};
            
            // Get window text with proper error checking
            int textLength = GetWindowTextA(hWnd, windowText, BUFFER_SIZE - 1);
            
            // Ensure null termination
            windowText[BUFFER_SIZE - 1] = '\0';
            
            // Thread-safe output
            if (textLength > 0)
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                std::cout << "Window created: " << windowText << std::endl;
            }
            else if (textLength == 0)
            {
                // Check if it's an error or just empty text
                DWORD error = GetLastError();
                if (error != ERROR_SUCCESS)
                {
                    std::lock_guard<std::mutex> lock(outputMutex);
                    std::cerr << "Failed to get window text. Error: " << error << std::endl;
                }
            }
        }
    }

    return CallNextHookEx(hHook, nCode, wParam, lParam);
}

int main()
{
    // Get current thread ID for thread-specific hook
    DWORD currentThreadId = GetCurrentThreadId();
    
    // Set the hook for current thread only (more reliable without DLL)
    // For system-wide hook, you would need a DLL
    hHook = SetWindowsHookEx(WH_SHELL, HookCallback, NULL, currentThreadId);

    if (hHook == NULL)
    {
        DWORD error = GetLastError();
        std::cerr << "Failed to set hook. Error code: " << error << std::endl;
        return 1;
    }

    std::cout << "Hook installed successfully. Monitoring current thread windows..." << std::endl;
    std::cout << "Press Ctrl+C to exit." << std::endl;

    // Message loop with proper error handling
    MSG msg;
    BOOL bRet;
    
    while ((bRet = GetMessage(&msg, NULL, 0, 0)) != 0)
    {
        if (bRet == -1)
        {
            // Handle error
            DWORD error = GetLastError();
            std::cerr << "GetMessage failed. Error code: " << error << std::endl;
            break;
        }
        else
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    // Unhook with error checking
    if (hHook != NULL)
    {
        if (!UnhookWindowsHookEx(hHook))
        {
            DWORD error = GetLastError();
            std::cerr << "Failed to unhook. Error code: " << error << std::endl;
            return 1;
        }
        else
        {
            std::cout << "Hook removed successfully." << std::endl;
        }
    }

    return 0;
}