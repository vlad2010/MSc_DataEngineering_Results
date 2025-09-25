#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

volatile bool handler_invoked = false;

void signal_handler(int signum) {
    handler_invoked = true;
}

int main() {
    struct sigaction sa, current_sa;
    sigset_t mask, oldmask;

    // Setup the signal handler
    sa.sa_handler = signal_handler;
    sa.sa_flags = SA_RESETHAND; // Reset the signal handler to default after first delivery
    sigemptyset(&sa.sa_mask);

    // Set the action for SIGINT
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        exit(10); // Exit with a unique non-zero code if sigaction fails to set the handler
    }

    // Block SIGINT and save current signal mask
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    if (sigprocmask(SIG_BLOCK, &mask, &oldmask) == -1) {
        exit(11); // Exit with a unique non-zero code if sigprocmask fails
    }

    pid_t pid = fork();
    if (pid == -1) {
        // Fork failed
        exit(12); // Exit with a unique non-zero code if fork fails
    } else if (pid == 0) {
        // Child process
        sleep(1); // Wait for 1 second before sending the signal
        kill(getppid(), SIGINT); // Send SIGINT to parent process
        exit(EXIT_SUCCESS);
    } else {
        // Parent process
        // Wait for the signal handler to be invoked
        while (!handler_invoked) {
            sigsuspend(&oldmask);
        }

        // Query the current signal handler for SIGINT
        if (sigaction(SIGINT, NULL, &current_sa) == -1) {
            exit(13); // Exit with a unique non-zero code if sigaction fails to retrieve the handler
        }

        // Check if the current signal handler is set to the default
        if (current_sa.sa_handler != SIG_DFL) {
            exit(14); // Exit with a unique non-zero code if the handler is not reset to default
        }

        exit(EXIT_SUCCESS);
    }
}