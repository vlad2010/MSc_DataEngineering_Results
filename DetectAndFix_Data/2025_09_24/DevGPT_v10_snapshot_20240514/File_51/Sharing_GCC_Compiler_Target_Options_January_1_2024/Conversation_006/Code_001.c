#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

void signal_handler(int signum) {
    // Exit with a non-zero code if called more than once, indicating SA_RESETHAND didn't work
    exit(EXIT_FAILURE);
}

int main() {
    struct sigaction sa;
    pid_t pid;

    // Setup the signal handler
    sa.sa_handler = signal_handler;
    sa.sa_flags = SA_RESETHAND; // Reset the signal handler to default after first delivery
    sigemptyset(&sa.sa_mask);

    // Set the action for SIGINT
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        exit(EXIT_FAILURE); // Exit with a non-zero code if sigaction fails
    }

    pid = fork();
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

        // If the program reaches this point, it means the signal handler wasn't triggered.
        // This is a failure in this context.
        exit(EXIT_FAILURE);
    }
}