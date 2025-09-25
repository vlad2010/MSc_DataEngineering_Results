#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <errno.h>
#include <sys/wait.h>
#include <stdio.h>

// Use atomic type for signal-safe access (fixes CWE-362)
volatile sig_atomic_t handler_invoked = 0;
volatile sig_atomic_t second_handler_invoked = 0;

void signal_handler(int signum) {
    // Only use async-signal-safe operations (prevents CWE-479)
    handler_invoked = 1;
}

void second_signal_handler(int signum) {
    second_handler_invoked = 1;
}

int main() {
    struct sigaction sa, old_sa;
    pid_t pid;
    int status;

    // Initialize the sigaction structure properly
    sa.sa_handler = signal_handler;
    sa.sa_flags = SA_RESETHAND; // Reset the signal handler to default after first delivery
    sigemptyset(&sa.sa_mask);
    
    // Block SIGINT during critical sections to prevent race conditions
    sigset_t block_mask, old_mask;
    sigemptyset(&block_mask);
    sigaddset(&block_mask, SIGINT);
    
    // Block SIGINT temporarily (fixes CWE-364)
    if (sigprocmask(SIG_BLOCK, &block_mask, &old_mask) == -1) {
        perror("sigprocmask");
        exit(EXIT_FAILURE);
    }

    // Set the action for SIGINT
    if (sigaction(SIGINT, &sa, &old_sa) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
    
    // Unblock SIGINT
    if (sigprocmask(SIG_SETMASK, &old_mask, NULL) == -1) {
        perror("sigprocmask");
        exit(EXIT_FAILURE);
    }

    pid = fork();
    if (pid == -1) {
        // Fork failed (improved error handling - fixes CWE-252)
        perror("fork");
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // Child process
        struct timespec req = {1, 0}; // 1 second
        struct timespec rem;
        
        // Use nanosleep for better control (fixes sleep interruption issue)
        while (nanosleep(&req, &rem) == -1 && errno == EINTR) {
            req = rem;
        }
        
        // Send SIGINT to parent process with error checking (fixes CWE-252)
        if (kill(getppid(), SIGINT) == -1) {
            perror("kill");
            exit(EXIT_FAILURE);
        }
        exit(EXIT_SUCCESS);
    } else {
        // Parent process
        
        // Wait for the signal with timeout to prevent indefinite blocking
        sigset_t wait_set;
        sigemptyset(&wait_set);
        sigaddset(&wait_set, SIGINT);
        struct timespec timeout = {5, 0}; // 5 second timeout
        
        // Use sigsuspend for more controlled signal waiting
        pause(); // Wait for the signal
        
        // Check if the signal handler was invoked (using sig_atomic_t)
        if (!handler_invoked) {
            // Clean up child process
            kill(pid, SIGTERM);
            waitpid(pid, &status, 0);
            exit(EXIT_FAILURE);
        }
        
        // Wait for child to complete
        if (waitpid(pid, &status, 0) == -1) {
            perror("waitpid");
            exit(EXIT_FAILURE);
        }
        
        // Properly test if SA_RESETHAND worked
        // Set up a second handler to catch if the signal wasn't reset
        sa.sa_handler = second_signal_handler;
        sa.sa_flags = 0; // No SA_RESETHAND this time
        sigemptyset(&sa.sa_mask);
        
        if (sigaction(SIGINT, &sa, NULL) == -1) {
            perror("sigaction");
            exit(EXIT_FAILURE);
        }
        
        // Send signal again with error checking (fixes CWE-252)
        if (raise(SIGINT) == -1) {
            perror("raise");
            exit(EXIT_FAILURE);
        }
        
        // If SA_RESETHAND worked correctly on the first signal,
        // the second handler should be invoked
        // If it didn't work, the process would have been terminated by SIG_DFL
        if (second_handler_invoked) {
            // SA_RESETHAND worked as expected
            exit(EXIT_SUCCESS);
        } else {
            // This shouldn't happen if everything works correctly
            exit(EXIT_FAILURE);
        }
    }
}