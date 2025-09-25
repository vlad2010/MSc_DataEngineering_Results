// Written by AI Assistant (hardened)
#include <climits>   // INT_MAX
#include <algorithm> // std::min, std::max
#include <string>

// Assumptions: external symbols exist:
// - volatile u32 kDown; #defines KEY_A/KEY_B
// - void switchMenu(Menu m); Menu mainMenu;
// - bool newArrayOnStart; void initArray();
// - Thread sortThread; bool threadIsRunning(Thread); int threadJoin(Thread, uint64_t timeout);
//   void threadFree(Thread); Thread threadCreate(void (*fn)(void*), void* arg, size_t stack, int prio, int cpuid, bool detached);
// - int prio; size_t STACKSIZE;
// - void insertionSort(void*);
// - containers ALGO_TEXT, DESCRIPTION_TEXT holding std::string, supporting size() and operator[].
// - printf available.

static void safePrintCentered(int row, const std::string& text, int centerCol /*e.g., 20*/, int minCol /*e.g., 1*/)
{
    // Compute half length safely in signed domain and clamp
    int halfLen = 0;
    if (text.size() >= static_cast<size_t>(INT_MAX)) {
        halfLen = INT_MAX / 2;
    } else {
        halfLen = static_cast<int>(text.size()) / 2;
    }
    int col = centerCol - halfLen;
    if (col < minCol) col = minCol;
    // Use %d for integers to match int arguments
    printf("\x1b[%d;%dH%s\n", row, col, text.c_str());
}

void algoMenuHandler()
{
    // Handle, if user pressed A
    if (kDown & KEY_A)
    {
        // Clean up any previous thread without blocking the UI indefinitely
        if (sortThread != nullptr)
        {
            if (!threadIsRunning(sortThread))
            {
                // Non-blocking or immediate join to reap the thread if needed, then free
                (void)threadJoin(sortThread, 0 /*non-blocking*/);
                threadFree(sortThread);
                sortThread = nullptr;
            }
            else
            {
                // A sort is already running; avoid blocking the UI (security: prevent DoS via indefinite join)
                safePrintCentered(22, "Sort is already running...", 20, 1);
                return;
            }
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

            // Clamp priority to avoid underflow/invalid values
            int startPrio = prio;
            if (startPrio > INT_MIN) {
                // Decrease priority by one if possible; ensure it stays within a sane range (example clamp 0..255 if applicable)
                startPrio = (startPrio > 0) ? (startPrio - 1) : startPrio;
            }

            Thread t = threadCreate(insertionSort, nullptr, STACKSIZE, startPrio, 1, false);
            if (t == nullptr)
            {
                safePrintCentered(22, "Failed to start sort thread", 20, 1);
                break;
            }
            sortThread = t;

            // Safe access to text arrays
            const std::string algoText = (ALGO_TEXT.size() > 1) ? ALGO_TEXT[1] : std::string("Insertion Sort");
            const std::string descText = (DESCRIPTION_TEXT.size() > 0) ? DESCRIPTION_TEXT[0] : std::string("");

            safePrintCentered(16, algoText, 20, 1);
            // Row 19, column 1; %s is fine for c_str()
            printf("\x1b[19;1H%s\n", descText.c_str());
            break;
        }
        // Repeat for other cases with the same safety patterns...
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
// Written by AI Assistant