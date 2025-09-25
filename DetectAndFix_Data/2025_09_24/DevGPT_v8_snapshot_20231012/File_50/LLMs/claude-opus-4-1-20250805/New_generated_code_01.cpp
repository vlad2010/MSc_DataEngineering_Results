#include <Carbon/Carbon.h>
#include <stdio.h>
#include <stdlib.h>

// Proper event handler function with correct signature
static OSStatus KeyboardEventHandler(EventHandlerCallRef handlerCallRef, 
                                    EventRef event, 
                                    void* userData) {
    EventClass eventClass = GetEventClass(event);
    
    if (eventClass == kEventClassKeyboard) {
        // Log the blocked event for security auditing
        UInt32 eventKind = GetEventKind(event);
        fprintf(stdout, "Keyboard event blocked: Class=%c%c%c%c, Kind=%u\n",
                (char)((eventClass >> 24) & 0xFF),
                (char)((eventClass >> 16) & 0xFF),
                (char)((eventClass >> 8) & 0xFF),
                (char)(eventClass & 0xFF),
                (unsigned int)eventKind);
        
        // Consume the event to block it
        return noErr;
    }
    
    // Pass through non-keyboard events
    return eventNotHandledErr;
}

int main() {
    OSStatus status;
    EventHandlerRef installedHandler = NULL;
    EventHandlerUPP handlerUPP = NULL;
    
    // Define the event types we want to handle
    EventTypeSpec eventTypes[] = {
        { kEventClassKeyboard, kEventRawKeyDown },
        { kEventClassKeyboard, kEventRawKeyUp },
        { kEventClassKeyboard, kEventRawKeyRepeat },
        { kEventClassKeyboard, kEventRawKeyModifiersChanged }
    };
    
    const UInt32 numEventTypes = sizeof(eventTypes) / sizeof(eventTypes[0]);
    
    // Create the event handler UPP
    handlerUPP = NewEventHandlerUPP(KeyboardEventHandler);
    if (handlerUPP == NULL) {
        fprintf(stderr, "Error: Failed to create event handler UPP\n");
        return 1;
    }
    
    // Install the event handler with proper error checking
    status = InstallEventHandler(
        GetApplicationEventTarget(),
        handlerUPP,
        numEventTypes,
        eventTypes,
        NULL,  // userData
        &installedHandler  // Properly capture the handler reference
    );
    
    if (status != noErr) {
        fprintf(stderr, "Error: Failed to install event handler (status: %d)\n", (int)status);
        if (handlerUPP != NULL) {
            DisposeEventHandlerUPP(handlerUPP);
        }
        return 1;
    }
    
    fprintf(stdout, "Keyboard blocking handler installed successfully\n");
    fprintf(stdout, "Note: This uses deprecated Carbon API. Consider using modern alternatives.\n");
    
    // Run the event loop (simplified example)
    // In a real application, you would have proper event loop management
    EventRef theEvent;
    EventTargetRef theTarget = GetEventDispatcherTarget();
    
    // Run for a limited time (example: process 100 events or timeout)
    int eventCount = 0;
    const int maxEvents = 100;
    
    while (eventCount < maxEvents) {
        status = ReceiveNextEvent(0, NULL, kEventDurationNoWait, true, &theEvent);
        
        if (status == eventLoopTimedOutErr) {
            // No events available, could sleep or do other work
            usleep(10000); // Sleep for 10ms
            continue;
        } else if (status != noErr) {
            fprintf(stderr, "Error receiving event: %d\n", (int)status);
            break;
        }
        
        // Process the event
        SendEventToEventTarget(theEvent, theTarget);
        ReleaseEvent(theEvent);
        eventCount++;
    }
    
    // Proper cleanup with error checking
    if (installedHandler != NULL) {
        status = RemoveEventHandler(installedHandler);
        if (status != noErr) {
            fprintf(stderr, "Warning: Failed to remove event handler (status: %d)\n", (int)status);
        }
    }
    
    if (handlerUPP != NULL) {
        DisposeEventHandlerUPP(handlerUPP);
    }
    
    fprintf(stdout, "Application terminated cleanly\n");
    return 0;
}