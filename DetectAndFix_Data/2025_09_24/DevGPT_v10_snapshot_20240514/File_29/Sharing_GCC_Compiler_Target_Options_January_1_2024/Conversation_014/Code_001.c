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

    // Setup the signal handler
    sa.sa_handler = signal_handler;
    sa.sa_flags = SA_RESETHAND; // Reset the signal handler to default after first delivery
    sigemptyset(&sa.sa_mask);

    // Set the action for SIGINT
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        exit(1); // Exit with a unique non-zero code if sigaction fails
    }

    // Sanity check: verify the current signal handler for SIGINT
    if (sigaction(SIGINT, NULL, &current_sa) == -1 || current_sa.sa_handler != signal_handler) {
        exit(2); // Exit with a unique non-zero code if the handler is not set as expected
    }

    pid_t pid = fork();
    if (pid == -1) {
        // Fork failed
        exit(3); // Exit with a unique non-zero code if fork fails
    } else if (pid == 0) {
        // Child process
        sleep(1); // Wait for 1 second before sending the signal
        kill(getppid(), SIGINT); // Send SIGINT to parent process
        exit(EXIT_SUCCESS);
    } else {
        // Parent process
        pause(); // Wait for the signal

        // Check if the signal handler was invoked
        if (!handler_invoked) {
            exit(4); // Exit with a unique non-zero code if handler was not invoked
        }

        // Query the current signal handler for SIGINT
        if (sigaction(SIGINT, NULL, &current_sa) == -1 || current_sa.sa_handler != SIG_DFL) {
            exit(5); // Exit with a unique non-zero code if the handler is not reset as expected
        }

        exit(EXIT_SUCCESS);
    }
}