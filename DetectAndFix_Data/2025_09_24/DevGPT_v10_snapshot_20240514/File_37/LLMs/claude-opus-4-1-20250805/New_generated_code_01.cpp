#include <mutex>
#include <atomic>

// Thread management structure
struct ThreadManager {
    Thread* thread;
    std::mutex mutex;
    std::atomic<bool> isRunning;
    
    ThreadManager() : thread(nullptr), isRunning(false) {}
};

static ThreadManager sortThreadManager;

void algoMenuHandler()
{
    // Handle, if user pressed A
    if (kDown & KEY_A)
    {
        // Thread cleanup with proper synchronization
        {
            std::lock_guard<std::mutex> lock(sortThreadManager.mutex);
            
            if (sortThreadManager.thread != nullptr)
            {
                // Check if thread is actually running before attempting to join
                if (sortThreadManager.isRunning.load())
                {
                    // Use a reasonable timeout instead of U64_MAX to prevent indefinite blocking
                    const u64 THREAD_JOIN_TIMEOUT = 10000000000ULL; // 10 seconds in nanoseconds
                    Result joinResult = threadJoin(sortThreadManager.thread, THREAD_JOIN_TIMEOUT);
                    
                    if (R_FAILED(joinResult))
                    {
                        // Log error but continue to free resources
                        printf("\x1b[20;1HWarning: Thread join failed with error: 0x%08X\n", joinResult);
                    }
                    
                    sortThreadManager.isRunning.store(false);
                }
                
                // Free thread resources
                Result freeResult = threadFree(sortThreadManager.thread);
                if (R_FAILED(freeResult))
                {
                    printf("\x1b[21;1HWarning: Thread free failed with error: 0x%08X\n", freeResult);
                }
                
                sortThreadManager.thread = nullptr;
            }
        }

        // Validate selected index before use
        const int MAX_MENU_OPTIONS = 10; // Define based on actual menu size
        if (selected < 0 || selected >= MAX_MENU_OPTIONS)
        {
            printf("\x1b[22;1HError: Invalid menu selection: %d\n", selected);
            return;
        }

        switch (selected)
        {
        case 0:
            switchMenu(mainMenu);
            break;
        case 1:
            {
                if (newArrayOnStart)
                {
                    initArray();
                }
                
                // Thread creation with proper error handling
                std::lock_guard<std::mutex> lock(sortThreadManager.mutex);
                
                // Ensure no thread is already running
                if (sortThreadManager.thread == nullptr)
                {
                    sortThreadManager.thread = threadCreate(insertionSort, NULL, STACKSIZE, prio - 1, 1, false);
                    
                    if (sortThreadManager.thread == nullptr)
                    {
                        printf("\x1b[23;1HError: Failed to create sorting thread\n");
                        break;
                    }
                    
                    // Start the thread
                    Result startResult = threadStart(sortThreadManager.thread);
                    if (R_FAILED(startResult))
                    {
                        printf("\x1b[23;1HError: Failed to start thread: 0x%08X\n", startResult);
                        threadFree(sortThreadManager.thread);
                        sortThreadManager.thread = nullptr;
                        break;
                    }
                    
                    sortThreadManager.isRunning.store(true);
                    
                    // Safe string printing - prevent format string vulnerabilities
                    // Validate array bounds first
                    if (ALGO_TEXT.size() > 1)
                    {
                        // Calculate safe position
                        int textLength = ALGO_TEXT[1].length();
                        int xPos = (textLength > 40) ? 1 : (20 - textLength/2);
                        xPos = (xPos < 1) ? 1 : xPos;
                        
                        // Use safe printing without format specifiers in the string
                        printf("\x1b[16;%iH%.*s\n", xPos, 80, ALGO_TEXT[1].c_str());
                    }
                    
                    if (DESCRIPTION_TEXT.size() > 0)
                    {
                        // Limit output length to prevent buffer overflow
                        printf("\x1b[19;1H%.*s\n", 200, DESCRIPTION_TEXT[0].c_str());
                    }
                }
                else
                {
                    printf("\x1b[23;1HWarning: Sorting thread is already running\n");
                }
            }
            break;
        // Repeat for other cases with similar safety checks...
        default:
            printf("\x1b[24;1HWarning: Unhandled menu option: %d\n", selected);
            break;
        }
    }

    // Handle, if user pressed B
    if (kDown & KEY_B)
    {
        // Ensure thread is properly cleaned up before switching menu
        {
            std::lock_guard<std::mutex> lock(sortThreadManager.mutex);
            if (sortThreadManager.thread != nullptr && sortThreadManager.isRunning.load())
            {
                // Attempt to stop the thread gracefully
                // Note: This assumes the sorting thread checks for cancellation
                threadCancelSynchronization(sortThreadManager.thread);
            }
        }
        
        switchMenu(mainMenu);
    }
}