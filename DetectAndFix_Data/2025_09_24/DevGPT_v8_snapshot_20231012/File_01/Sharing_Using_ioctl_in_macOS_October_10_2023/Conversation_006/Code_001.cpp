#include <iostream>
#include <Carbon/Carbon.h>

double rightClickDown = 0.0;

CGEventRef tapCallback(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void* refcon) {
    double now = CFAbsoluteTimeGetCurrent();
    if (type == kCGEventRightMouseDown) {
        rightClickDown = now;
    } else if (type == kCGEventRightMouseUp) {
        if (now - rightClickDown >= 1.0) {
            exit(0); // Exits the application if right-click is held for 1 second
        }
    }
    return nullptr; // Consumes the event, preventing it from being processed further
}

int main() {
    CFMachPortRef eventTap;
    CGEventMask mask = kCGEventMaskForAllEvents;
    
    eventTap = CGEventTapCreate(kCGHIDEventTap, kCGHeadInsertEventTap,
        kCGEventTapOptionDefault, mask, tapCallback, nullptr);

    if (!eventTap) {
        std::cerr << "Failed to create event tap." << std::endl;
        return 1;
    }

    CFRunLoopSourceRef runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, eventTap, 0);
    CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, kCFRunLoopCommonModes);
    CGEventTapEnable(eventTap, true);

    CFRunLoopRun();

    // Clean up
    CFRelease(runLoopSource);
    CFRelease(eventTap);

    return 0;
}
