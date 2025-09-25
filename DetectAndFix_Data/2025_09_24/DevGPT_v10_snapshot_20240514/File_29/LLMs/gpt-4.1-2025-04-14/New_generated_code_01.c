#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdio.h>
#include <errno.h>

volatile sig_atomic_t handler_invoked = 0; // FIX: Use sig_atomic_t for signal safety

void signal_handler(int signum) {
    handler_invoked = 1; // FIX: Only use async-signal-safe operations
}

int main() {
    struct sigaction sa;
    sigset_t block_mask, old_mask;

    // Block SIGINT before installing handler to avoid race condition
    sigemptyset(&block_mask);
    sigaddset(&block_mask, SIGINT);
    if (sigprocmask(SIG_BLOCK, &block_mask, &old_mask) == -1) {
        perror("sigprocmask");
        exit(EXIT_FAILURE);
    }

    // Setup the signal handler
    sa.sa_handler = signal_handler;
    sa.sa_flags = SA_RESETHAND; // Reset the signal handler to default after first delivery
    sigemptyset(&sa.sa_mask);

    // Set the action for SIGINT
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE); // Exit with a non-zero code if sigaction fails
    }

    pid_t pid = fork();
    if (pid == -1) {
        // Fork failed
        perror("fork");
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // Child process
        sleep(1); // Wait for 1 second before sending the signal
        if (kill(getppid(), SIGINT) == -1) { // FIX: Check kill() return value
            perror("kill");
            exit(EXIT_FAILURE);
        }
        exit(EXIT_SUCCESS);
    } else {
        // Parent process

        // Unblock SIGINT and wait for it atomically
        sigset_t suspend_mask;
        sigemptyset(&suspend_mask);
        // Wait for signal with sigsuspend to avoid race condition
        while (!handler_invoked) {
            if (sigsuspend(&suspend_mask) == -1 && errno != EINTR) {
                perror("sigsuspend");
                exit(EXIT_FAILURE);
            }
        }

        // Restore old signal mask
        if (sigprocmask(SIG_SETMASK, &old_mask, NULL) == -1) {
            perror("sigprocmask restore");
            exit(EXIT_FAILURE);
        }

        // Check if the signal handler was invoked
        if (!handler_invoked) {
            fprintf(stderr, "Signal handler was not invoked\n");
            exit(EXIT_FAILURE);
        }

        // Set up a second signal handler to test if the first was reset
        sa.sa_handler = SIG_DFL;
        if (sigaction(SIGINT, &sa, NULL) == -1) {
            perror("sigaction reset");
            exit(EXIT_FAILURE); // Exit with a non-zero code if sigaction fails
        }

        // Trigger SIGINT again to test if the handler has been reset
        raise(SIGINT);

        // If the program reaches this point, the handler was not reset as expected
        fprintf(stderr, "Handler was not reset as expected\n");
        exit(EXIT_FAILURE);
    }
}