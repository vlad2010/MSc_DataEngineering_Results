#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <errno.h>
#include <sys/wait.h>

// Use atomic for signal-safe access (fixes CWE-362)
volatile sig_atomic_t handler_invoked = 0;

void signal_handler(int signum) {
    // Use sig_atomic_t for signal-safe operations
    handler_invoked = 1;
}

int main() {
    struct sigaction sa, current_sa;
    int pipefd[2];
    char buf;
    sigset_t mask, oldmask;
    pid_t pid;
    int status;

    // Create a pipe for synchronization
    if (pipe(pipefd) == -1) {
        exit(18);
    }

    // Setup the signal handler
    sa.sa_handler = signal_handler;
    sa.sa_flags = SA_RESETHAND; // Reset the signal handler to default after first delivery
    sigemptyset(&sa.sa_mask);

    // Block SIGINT before setting up handler to prevent race condition (CWE-479)
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    if (sigprocmask(SIG_BLOCK, &mask, &oldmask) == -1) {
        exit(19);
    }

    // Set the action for SIGINT
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        exit(10);
    }

    // Sanity check: verify the current signal handler for SIGINT
    if (sigaction(SIGINT, NULL, &current_sa) == -1) {
        exit(11);
    }
    if (current_sa.sa_handler != signal_handler) {
        exit(12);
    }
    if ((current_sa.sa_flags & SA_RESETHAND) != SA_RESETHAND) {
        exit(17);
    }

    pid = fork();
    if (pid == -1) {
        // Fork failed
        exit(13);
    } else if (pid == 0) {
        // Child process
        // Restore signal mask in child
        sigprocmask(SIG_SETMASK, &oldmask, NULL);
        
        // FIXED: Close correct end of pipe (fixes CWE-667)
        if (close(pipefd[1]) == -1) {  // Close unused write end
            exit(20);
        }

        // Wait for parent to be ready - check return value (fixes CWE-252)
        ssize_t bytes_read = read(pipefd[0], &buf, 1);
        if (bytes_read != 1) {
            exit(21);
        }
        
        if (close(pipefd[0]) == -1) {
            exit(22);
        }

        // Send SIGINT to parent process
        if (kill(getppid(), SIGINT) == -1) {
            exit(23);
        }
        exit(EXIT_SUCCESS);
    } else {
        // Parent process
        // FIXED: Close correct end of pipe (fixes CWE-667)
        if (close(pipefd[0]) == -1) {  // Close unused read end
            exit(24);
        }

        // Signal child that parent is ready - check return value (fixes CWE-252)
        ssize_t bytes_written = write(pipefd[1], "r", 1);
        if (bytes_written != 1) {
            exit(25);
        }
        
        if (close(pipefd[1]) == -1) {
            exit(26);
        }

        // FIXED: Use sigsuspend instead of pause to avoid race condition (fixes CWE-479)
        // Atomically unblock SIGINT and wait for signal
        sigemptyset(&mask);
        while (!handler_invoked) {
            if (sigsuspend(&mask) == -1 && errno != EINTR) {
                exit(27);
            }
        }

        // Check if the signal handler was invoked
        if (!handler_invoked) {
            exit(14);
        }

        // Query the current signal handler for SIGINT
        if (sigaction(SIGINT, NULL, &current_sa) == -1) {
            exit(15);
        }
        if (current_sa.sa_handler != SIG_DFL) {
            exit(16);
        }

        // Wait for child to prevent zombie process
        if (waitpid(pid, &status, 0) == -1) {
            exit(28);
        }

        exit(EXIT_SUCCESS);
    }
}