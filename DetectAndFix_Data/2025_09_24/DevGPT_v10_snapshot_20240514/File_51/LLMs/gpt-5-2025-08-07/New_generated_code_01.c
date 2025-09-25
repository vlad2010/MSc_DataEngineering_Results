#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>

static volatile sig_atomic_t handler_invoked = 0;

static void signal_handler(int signum) {
    (void)signum;
    handler_invoked = 1;
}

static int set_cloexec(int fd) {
    int flags = fcntl(fd, F_GETFD);
    if (flags == -1) return -1;
    return fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
}

int main(void) {
    struct sigaction sa, current_sa;
    int pipefd[2];
    char buf;
    sigset_t block_set, old_set, wait_set;

    if (pipe(pipefd) == -1) {
        exit(18); // pipe creation failed
    }
    // Hardening: avoid FD leaks across exec
    (void)set_cloexec(pipefd[0]);
    (void)set_cloexec(pipefd[1]);

    // Setup the signal handler
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sa.sa_flags = SA_RESETHAND; // Reset the signal handler to default after first delivery
    if (sigemptyset(&sa.sa_mask) == -1) {
        exit(24); // unique non-zero code if sigemptyset fails
    }

    if (sigaction(SIGINT, &sa, NULL) == -1) {
        exit(10); // sigaction set failed
    }

    // Sanity check: verify the current signal handler for SIGINT
    if (sigaction(SIGINT, NULL, &current_sa) == -1) {
        exit(11); // retrieve failed
    }
    if (current_sa.sa_handler != signal_handler) {
        exit(12); // handler doesn't match
    }
    if ((current_sa.sa_flags & SA_RESETHAND) != SA_RESETHAND) {
        exit(17); // SA_RESETHAND not set
    }

    // Block SIGINT to avoid lost-wakeup race; use sigsuspend to wait atomically later
    if (sigemptyset(&block_set) == -1) {
        exit(25);
    }
    if (sigaddset(&block_set, SIGINT) == -1) {
        exit(26);
    }
    if (sigprocmask(SIG_BLOCK, &block_set, &old_set) == -1) {
        exit(19); // block failed
    }

    pid_t pid = fork();
    if (pid == -1) {
        // Fork failed
        exit(13);
    } else if (pid == 0) {
        // Child process
        // Close unused write end; keep read end to wait for readiness signal
        if (close(pipefd[1]) == -1) {
            _exit(27);
        }

        // Wait for parent to be ready: read exactly 1 byte
        ssize_t n;
        do {
            n = read(pipefd[0], &buf, 1);
        } while (n == -1 && errno == EINTR);
        if (n != 1) {
            // Either error or EOF; synchronization failed
            _exit(20);
        }

        // Done with pipe read end
        (void)close(pipefd[0]);

        // Send SIGINT to parent process
        pid_t ppid = getppid();
        if (ppid > 1) {
            if (kill(ppid, SIGINT) == -1) {
                _exit(21);
            }
        }
        _exit(EXIT_SUCCESS);
    } else {
        // Parent process
        // Close unused read end; write a byte to signal readiness
        if (close(pipefd[0]) == -1) {
            exit(28);
        }

        ssize_t w;
        do {
            w = write(pipefd[1], "r", 1);
        } while (w == -1 && errno == EINTR);
        if (w != 1) {
            // Write failed
            exit(23);
        }
        // Close write end; child has enough to proceed
        (void)close(pipefd[1]);

        // Prepare wait mask: old_set with SIGINT unblocked
        wait_set = old_set;
        if (sigdelset(&wait_set, SIGINT) == -1) {
            exit(29);
        }

        // Atomically wait for SIGINT without race (CWE-362 fix)
        while (!handler_invoked) {
            sigsuspend(&wait_set); // returns -1 with errno=EINTR when a signal is delivered
        }

        // Restore original signal mask
        if (sigprocmask(SIG_SETMASK, &old_set, NULL) == -1) {
            exit(30);
        }

        // Check if the signal handler was invoked
        if (!handler_invoked) {
            exit(14);
        }

        // Query the current signal handler for SIGINT (should be default due to SA_RESETHAND)
        if (sigaction(SIGINT, NULL, &current_sa) == -1) {
            exit(15);
        }
        if (current_sa.sa_handler != SIG_DFL) {
            exit(16);
        }

        // Reap child to avoid zombie (CWE-772)
        int status;
        pid_t wpid;
        do {
            wpid = waitpid(pid, &status, 0);
        } while (wpid == -1 && errno == EINTR);
        if (wpid == -1) {
            exit(22);
        }

        exit(EXIT_SUCCESS);
    }
}