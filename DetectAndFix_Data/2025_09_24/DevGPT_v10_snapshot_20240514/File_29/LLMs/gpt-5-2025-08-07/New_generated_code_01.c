#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <sys/wait.h>
#include <errno.h>

volatile sig_atomic_t handler_invoked = 0;

void signal_handler(int signum) {
    (void)signum; // unused
    handler_invoked = 1; // async-signal-safe
}

int main(void) {
    struct sigaction sa;
    sigset_t block_mask, old_mask, suspend_mask;

    // Block SIGINT to avoid race between setting handler and waiting
    if (sigemptyset(&block_mask) == -1) {
        exit(EXIT_FAILURE);
    }
    if (sigaddset(&block_mask, SIGINT) == -1) {
        exit(EXIT_FAILURE);
    }
    if (sigprocmask(SIG_BLOCK, &block_mask, &old_mask) == -1) {
        exit(EXIT_FAILURE);
    }

    // Install handler with SA_RESETHAND
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sa.sa_flags = SA_RESETHAND;
    if (sigemptyset(&sa.sa_mask) == -1) {
        exit(EXIT_FAILURE);
    }
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        exit(EXIT_FAILURE);
    }

    pid_t pid = fork();
    if (pid == -1) {
        // Fork failed
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // Child process
        sleep(1); // Wait briefly before sending the signal
        pid_t ppid = getppid();
        if (ppid > 1) {
            if (kill(ppid, SIGINT) == -1) {
                _exit(EXIT_FAILURE);
            }
        }
        _exit(EXIT_SUCCESS);
    } else {
        // Parent process: atomically wait for SIGINT
        suspend_mask = old_mask;
        sigdelset(&suspend_mask, SIGINT); // ensure SIGINT is unblocked during sigsuspend

        while (!handler_invoked) {
            // Wait until SIGINT is delivered; this unblocks SIGINT only for the wait
            sigsuspend(&suspend_mask);
        }

        // Restore original signal mask
        if (sigprocmask(SIG_SETMASK, &old_mask, NULL) == -1) {
            exit(EXIT_FAILURE);
        }

        // Reap child to avoid zombie
        int status;
        (void)waitpid(pid, &status, 0);

        // Verify that the handler ran
        if (!handler_invoked) {
            exit(EXIT_FAILURE);
        }

        // Now reset SIGINT to default; reinitialize struct to avoid stale flags
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = SIG_DFL;
        sa.sa_flags = 0;
        if (sigemptyset(&sa.sa_mask) == -1) {
            exit(EXIT_FAILURE);
        }
        if (sigaction(SIGINT, &sa, NULL) == -1) {
            exit(EXIT_FAILURE);
        }

        // Trigger SIGINT again; this will terminate the process by default
        // This line is expected to not return.
        raise(SIGINT);

        // If we reached here, something went wrong
        exit(EXIT_FAILURE);
    }
}