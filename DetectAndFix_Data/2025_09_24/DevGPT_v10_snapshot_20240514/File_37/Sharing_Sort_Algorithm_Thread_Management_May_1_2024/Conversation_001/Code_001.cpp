// Written by AI Assistant
void algoMenuHandler()
{
    // Handle, if user pressed A
    if (kDown & KEY_A)
    {
        if (sortThread != nullptr && threadIsRunning(sortThread))
        {
            threadJoin(sortThread, U64_MAX); // Wait indefinitely for the thread to finish
            threadFree(sortThread);
            sortThread = nullptr;
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
            sortThread = threadCreate(insertionSort, NULL, STACKSIZE, prio - 1, 1, false);
            printf("\x1b[16;%iH%s\n", (20 - ALGO_TEXT[1].length()/2), ALGO_TEXT[1].c_str());
            printf("\x1b[19;1H%s\n", DESCRIPTION_TEXT[0].c_str());
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
// Written by AI Assistant