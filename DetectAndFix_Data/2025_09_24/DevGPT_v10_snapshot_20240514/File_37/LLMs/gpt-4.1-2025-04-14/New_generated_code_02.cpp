#include <cstdio>
#include <mutex>
#include <string>

// Assume these are defined elsewhere
extern int kDown;
extern int KEY_A, KEY_B;
extern void* sortThread;
extern bool threadIsRunning(void*);
extern int threadJoin(void*, unsigned long long);
extern int threadFree(void*);
extern void* threadCreate(void (*func)(void*), void*, size_t, int, int, bool);
extern void switchMenu(void*);
extern void* mainMenu;
extern int selected;
extern bool newArrayOnStart;
extern void initArray();
extern int STACKSIZE;
extern int prio;
extern std::string ALGO_TEXT[];
extern std::string DESCRIPTION_TEXT[];

// Mutex for thread synchronization
std::mutex sortThreadMutex;

void algoMenuHandler()
{
    // Handle, if user pressed A
    if (kDown & KEY_A)
    {
        // Synchronize access to sortThread
        {
            std::lock_guard<std::mutex> lock(sortThreadMutex);
            if (sortThread != nullptr && threadIsRunning(sortThread))
            {
                if (threadJoin(sortThread, ULLONG_MAX) != 0) // Check return value
                {
                    fprintf(stderr, "Error: threadJoin failed\n");
                    // Handle error appropriately
                }
                if (threadFree(sortThread) != 0) // Check return value
                {
                    fprintf(stderr, "Error: threadFree failed\n");
                    // Handle error appropriately
                }
                sortThread = nullptr;
            }
        }

        switch (selected)
        {
        case 0:
            switchMenu(mainMenu);
            break;
        case 1:
            if (newArrayOnStart)
            {
                initArray();
            }
            {
                std::lock_guard<std::mutex> lock(sortThreadMutex);
                sortThread = threadCreate(insertionSort, NULL, STACKSIZE, prio - 1, 1, false);
            }
            // Sanitize and limit format string usage
            if (ALGO_TEXT[1].find('%') == std::string::npos && !ALGO_TEXT[1].empty())
            {
                printf("\x1b[16;%iH%s\n", (20 - static_cast<int>(ALGO_TEXT[1].length())/2), ALGO_TEXT[1].c_str());
            }
            else
            {
                printf("\x1b[16;1H[Invalid Algorithm Text]\n");
            }
            if (DESCRIPTION_TEXT[0].find('%') == std::string::npos && !DESCRIPTION_TEXT[0].empty())
            {
                printf("\x1b[19;1H%s\n", DESCRIPTION_TEXT[0].c_str());
            }
            else
            {
                printf("\x1b[19;1H[Invalid Description]\n");
            }
            break;
        // Repeat for other cases...
        default:
            break;
        }
    }

    // Handle, if user pressed B
    if (kDown & KEY_B)
    {
        switchMenu(mainMenu);
    }
}