Below is a security-focused review and a hardened version of your code. The most relevant issues and fixes:

Key security issues and CWE references
- Unchecked return value from inet_pton may cause unintended bind to 0.0.0.0 (INADDR_ANY)
  - If inet_pton fails, addr.sin_addr stays zeroed from memset, so bind() will listen on all interfaces, potentially exposing the service unexpectedly.
  - CWE-252: Unchecked Return Value; CWE-703: Improper Check or Handling of Exceptional Conditions; also leads to unintended exposure (CWE-284: Improper Access Control).
- File descriptor not closed on all paths
  - You neither close() the socket on success nor on error paths after creation. In long-running or library code this causes descriptor leaks and can be exploited for resource-exhaustion DoS.
  - CWE-775: Missing Release of File Descriptor or Handle after Effective Lifetime; CWE-404: Improper Resource Shutdown or Release.
- No close-on-exec protection
  - Without SOCK_CLOEXEC/FD_CLOEXEC, the socket can leak into child processes across exec(), creating unintended access channels or resource retention.
  - CWE-404/CWE-775 (resource lifetime) and CWE-664: Improper Control of a Resource Through its Lifetime.
- Unnecessary use of SO_REUSEADDR on unicast UDP
  - For unicast UDP, SO_REUSEADDR is typically unnecessary and can facilitate port sharing/confusion depending on OS semantics. Prefer not to enable it unless there is a clear, justified need (e.g., multicast receivers).
  - CWE-693: Protection Mechanism Failure (misconfiguration that weakens isolation), conceptually related to CWE-284.
- Hard-coded IP/port (low severity, maintainability)
  - Hard-coding security-relevant configuration can lead to accidental exposure when code is reused in different environments.
  - CWE-547: Use of Hard-coded, Security-relevant Constants.

Hardened code
- Checks inet_pton return and fails closed (does not bind if address is invalid).
- Uses SOCK_CLOEXEC where available, with a fallback fcntl(FD_CLOEXEC).
- Properly closes the socket on all exit paths.
- Removes SO_REUSEADDR for unicast UDP (commented guidance on when and how to use it safely).
- Prints the actual bound address and port using inet_ntop/ntohs to avoid mismatches.
- Uses sizeof(enable) in setsockopt if you ever re-enable it.

Single code fragment with fixes:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

static int set_cloexec(int fd) {
    int flags = fcntl(fd, F_GETFD);
    if (flags == -1) return -1;
    if (fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == -1) return -1;
    return 0;
}

int main(void)
{
    int sockfd = -1;
    struct sockaddr_in addr;
    const char *bind_ip = "192.168.1.2";
    const unsigned short bind_port = 12345;

    // Create a socket with close-on-exec if available
#if defined(SOCK_CLOEXEC)
    sockfd = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
#else
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
#endif
    if (sockfd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

#ifndef SOCK_CLOEXEC
    // Fallback: ensure FD is not inherited across exec
    if (set_cloexec(sockfd) < 0) {
        perror("fcntl(FD_CLOEXEC)");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
#endif

    // For unicast UDP, avoid SO_REUSEADDR unless you have a specific, justified need
    // (e.g., multicast receivers, rapid restart scenarios in certain OSes).
    // If you must enable it, uncomment below and understand OS-specific semantics.
    /*
    int enable = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0) {
        perror("setsockopt(SO_REUSEADDR)");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    */

    // Bind the socket to a specific IP address and port
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(bind_port);

    int ip_rc = inet_pton(AF_INET, bind_ip, &addr.sin_addr);
    if (ip_rc == 0) {
        fprintf(stderr, "Invalid IPv4 address: %s\n", bind_ip);
        close(sockfd);
        exit(EXIT_FAILURE);
    } else if (ip_rc < 0) {
        perror("inet_pton");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    if (bind(sockfd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    char bound_ip[INET_ADDRSTRLEN] = {0};
    if (inet_ntop(AF_INET, &addr.sin_addr, bound_ip, sizeof(bound_ip)) == NULL) {
        // Non-fatal, but reportable
        strncpy(bound_ip, "(unknown)", sizeof(bound_ip) - 1);
    }

    printf("Socket successfully bound to %s:%u\n", bound_ip, (unsigned)ntohs(addr.sin_port));

    // Now you can use the socket...

    // Clean shutdown: close socket when done
    if (close(sockfd) < 0) {
        perror("close");
        // Don't exit with failure at program end solely for close error
    }

    return 0;
}
```

Notes
- If you truly need fast restarts or multicast, document why SO_REUSEADDR is required and consider additional safeguards:
  - For Linux UDP load sharing, prefer SO_REUSEPORT with consistent group membership and proper permissions.
  - For Windows, consider SO_EXCLUSIVEADDRUSE instead of SO_REUSEADDR to prevent port hijacking.
- Consider loading IP/port from configuration/environment (validated) rather than hard-coding (CWE-547).
- Compile with hardening flags and warnings: -Wall -Wextra -Werror -O2 -D_FORTIFY_SOURCE=2 and use modern tooling (ASAN/UBSAN) during testing.