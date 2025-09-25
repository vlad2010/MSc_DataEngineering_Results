// Build with: clang -std=c11 -Wall -Wextra -framework ApplicationServices -framework CoreFoundation keyboard_blocker.c -o keyboard_blocker
#include <ApplicationServices/ApplicationServices.h>
#include <CoreFoundation/CoreFoundation.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    CFMachPortRef tap;
} TapContext;

static CGEventRef keyboard_blocker_callback(CGEventTapProxy proxy,
                                            CGEventType type,
                                            CGEventRef event,
                                            void *refcon) {
    TapContext *ctx = (TapContext *)refcon;

    // Handle tap disable notifications per docs.
    if (type == kCGEventTapDisabledByTimeout || type == kCGEventTapDisabledByUserInput) {
        if (ctx && ctx->tap) {
            CGEventTapEnable(ctx->tap, true);
        }
        // Let the event pass in this edge case to avoid deadlocking input during re-enable.
        return event;
    }

    // If it's not a key-related event, pass it through unchanged.
    switch (type) {
        case kCGEventKeyDown:
        case kCGEventKeyUp:
        case kCGEventFlagsChanged:
            // Returning NULL causes the event to be dropped (blocked).
            return NULL;
        default:
            return event;
    }
}

static bool ensure_accessibility_permission(void) {
    // Prompt the user for Accessibility permission if not granted.
    CFStringRef kPromptKey = kAXTrustedCheckOptionPrompt;
    const void *keys[] = { kPromptKey };
    const void *vals[] = { kCFBooleanTrue };
    CFDictionaryRef opts = CFDictionaryCreate(kCFAllocatorDefault,
                                              keys, vals,
                                              1,
                                              &kCFTypeDictionaryKeyCallBacks,
                                              &kCFTypeDictionaryValueCallBacks);
    bool trusted = AXIsProcessTrustedWithOptions(opts);
    if (opts) CFRelease(opts);
    return trusted;
}

int main(void) {
    // Require Accessibility permission for global keyboard interception.
    if (!ensure_accessibility_permission()) {
        fprintf(stderr, "Accessibility (AX) permission is required to block keyboard events. "
                        "Grant permission in System Settings -> Privacy & Security -> Accessibility.\n");
        return EXIT_FAILURE;
    }

    // Define a mask for keyboard-related events.
    CGEventMask mask = (CGEventMaskBit(kCGEventKeyDown) |
                        CGEventMaskBit(kCGEventKeyUp) |
                        CGEventMaskBit(kCGEventFlagsChanged));

    TapContext ctx = {0};

    // Create the event tap at the session level so we can intercept user input.
    ctx.tap = CGEventTapCreate(kCGSessionEventTap,
                               kCGHeadInsertEventTap,
                               kCGEventTapOptionDefault,  // not listenOnly: we intend to modify (drop) events
                               mask,
                               keyboard_blocker_callback,
                               &ctx);
    if (ctx.tap == NULL) {
        fprintf(stderr, "Failed to create event tap. Ensure the process is trusted for Accessibility and not sandboxed without proper entitlement.\n");
        return EXIT_FAILURE;
    }

    // Create a run loop source and add to the current run loop.
    CFRunLoopSourceRef source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, ctx.tap, 0);
    if (source == NULL) {
        fprintf(stderr, "Failed to create run loop source for event tap.\n");
        CFRelease(ctx.tap);
        return EXIT_FAILURE;
    }

    CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes);

    // Enable the event tap explicitly (should be enabled by default, but be explicit).
    CGEventTapEnable(ctx.tap, true);

    fprintf(stdout, "Keyboard events are now being blocked. Press Ctrl+C to exit.\n");

    // Run the CFRunLoop to process events.
    CFRunLoopRun();

    // Cleanup on exit (normally reached when CFRunLoopStop is called elsewhere).
    CGEventTapEnable(ctx.tap, false);
    CFRunLoopRemoveSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes);
    CFRelease(source);
    CFRelease(ctx.tap);

    return EXIT_SUCCESS;
}