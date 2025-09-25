#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

volatile bool handler_invoked = false;

void signal_handler(int signum) {
    handler_invoked = true;
}

int main() {
    struct sigaction sa;

    // Setup the signal handler
    sa.sa_handler = signal_handler;
    sa.sa_flags = SA_RESETHAND; // Reset the signal handler to default after first delivery
    sigemptyset(&sa.sa_mask);

    // Set the action for SIGINT
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        exit(EXIT_FAILURE); // Exit with a non-zero code if sigaction fails
    }

    pid_t pid = fork();
    if (pid == -1) {
        // Fork failed
        exit(EXIT_FAILURE);
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
            exit(EXIT_FAILURE);
        }

        // Query the current signal handler for SIGINT
        struct sigaction current_sa;
        if (sigaction(SIGINT, NULL, &current_sa) == -1) {
            exit(EXIT_FAILURE); // Exit with a non-zero code if sigaction query fails
        }

        // Check if the current signal handler is set to the default
        if (current_sa.sa_handler != SIG_DFL) {
            exit(EXIT_FAILURE);
        }

        exit(EXIT_SUCCESS);
    }
}