#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdio.h>
#include <errno.h>

volatile sig_atomic_t handler_invoked = 0; // FIX 1: Use sig_atomic_t

void signal_handler(int signum) {
    handler_invoked = 1; // FIX 1: Only assign sig_atomic_t
}

int main() {
    struct sigaction sa, current_sa;
    int pipefd[2];
    char buf;

    // Create a pipe for synchronization
    if (pipe(pipefd) == -1) {
        perror("pipe");
        exit(18);
    }

    // Setup the signal handler
    sa.sa_handler = signal_handler;
    sa.sa_flags = SA_RESETHAND;
    sigemptyset(&sa.sa_mask);

    if (sigaction(SIGINT, &sa, NULL) == -1) {
        perror("sigaction set");
        close(pipefd[0]);
        close(pipefd[1]);
        exit(10);
    }

    if (sigaction(SIGINT, NULL, &current_sa) == -1) {
        perror("sigaction get");
        close(pipefd[0]);
        close(pipefd[1]);
        exit(11);
    }
    if (current_sa.sa_handler != signal_handler) {
        close(pipefd[0]);
        close(pipefd[1]);
        exit(12);
    }
    if ((current_sa.sa_flags & SA_RESETHAND) != SA_RESETHAND) {
        close(pipefd[0]);
        close(pipefd[1]);
        exit(17);
    }

    pid_t pid = fork();
    if (pid == -1) {
        // Fork failed
        close(pipefd[0]);
        close(pipefd[1]);
        exit(13);
    } else if (pid == 0) {
        // Child process
        close(pipefd[1]); // FIX 2: Close write end in child

        // Wait for parent to be ready
        ssize_t r = read(pipefd[0], &buf, 1); // FIX 3: Check return value
        if (r != 1) {
            perror("child read");
            close(pipefd[0]);
            exit(EXIT_FAILURE);
        }
        close(pipefd[0]);

        // Send SIGINT to parent process
        if (kill(getppid(), SIGINT) == -1) {
            perror("kill");
            exit(EXIT_FAILURE);
        }
        exit(EXIT_SUCCESS);
    } else {
        // Parent process
        close(pipefd[0]); // FIX 2: Close read end in parent

        // Signal child that parent is ready
        ssize_t w = write(pipefd[1], "r", 1); // FIX 3: Check return value
        if (w != 1) {
            perror("parent write");
            close(pipefd[1]);
            exit(EXIT_FAILURE);
        }
        close(pipefd[1]);

        pause(); // Wait for the signal

        // Check if the signal handler was invoked
        if (!handler_invoked) {
            exit(14);
        }

        if (sigaction(SIGINT, NULL, &current_sa) == -1) {
            exit(15);
        }
        if (current_sa.sa_handler != SIG_DFL) {
            exit(16);
        }

        exit(EXIT_SUCCESS);
    }
}