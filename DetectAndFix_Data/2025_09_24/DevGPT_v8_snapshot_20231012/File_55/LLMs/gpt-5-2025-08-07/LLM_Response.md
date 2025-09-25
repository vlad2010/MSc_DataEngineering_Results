Below is a security-focused review and a corrected version of your code.

Key security issues and fixes
- SO_BINDTODEVICE not null-terminated (CWE-170: Improper Null Termination)
  The kernel expects a null-terminated interface name and the optlen must include the terminating null. Passing strlen(interface) omits the null terminator and can cause failure or unexpected behavior. Fix: pass strlen(name) + 1, and ensure length <= IFNAMSIZ - 1.

- Input validation missing for IP address and interface length (CWE-20: Improper Input Validation)
  inet_pton’s return value is not checked; on failure your zero-initialized sockaddr may result in binding to 0.0.0.0 (all interfaces), potentially exposing the service more broadly than intended. Also, the interface name must be validated against IFNAMSIZ. Fix: validate both and bail out on errors.

- Missing release of file descriptor on error/success paths (CWE-775: Missing Release of File Descriptor or Handle after Effective Lifetime)
  Several exit paths leak the socket. Even though the OS will clean up on process exit, ensuring proper close() is important for correctness and future code changes. Fix: close the socket on all paths.

- Execution with unnecessary privileges (CWE-250)
  SO_BINDTODEVICE requires CAP_NET_ADMIN (or root). If the program is started with elevated privileges, they should be dropped as soon as the privileged operation is complete. Fix: after setsockopt succeeds, drop to the real UID/GID (if possible). Prefer granting only CAP_NET_ADMIN to the binary (e.g., setcap 'cap_net_admin=+ep') and do not run as full root.

- Hardening: avoid file descriptor leakage across exec
  Use SOCK_CLOEXEC when creating the socket to prevent accidental fd inheritance if the process execs another program later.

Fixed code (single fragment)
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <grp.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <linux/if.h>  // IFNAMSIZ

static int drop_privileges_to_real_ids(void) {
    uid_t ruid = getuid();
    gid_t rgid = getgid();
    uid_t euid = geteuid();

    // If not running with elevated privileges, nothing to drop.
    if (euid != 0) {
        return 0;
    }

    // Drop supplementary groups
    if (setgroups(0, NULL) != 0) {
        perror("setgroups");
        return -1;
    }

    // Drop GID first
    if (setresgid(rgid, rgid, rgid) != 0) {
        perror("setresgid");
        return -1;
    }

    // Drop UID next
    if (setresuid(ruid, ruid, ruid) != 0) {
        perror("setresuid");
        return -1;
    }

    return 0;
}

int main(void) {
    int sockfd = -1;
    struct sockaddr_in addr;
    const char *ip = "192.168.1.2";
    const char *interface = "eth0";  // Validate length and ensure null-termination
    char ifname[IFNAMSIZ];

    // Validate interface name length to avoid truncation/misuse (CWE-20)
    size_t iflen = strnlen(interface, IFNAMSIZ);
    if (iflen == 0 || iflen >= IFNAMSIZ) {
        fprintf(stderr, "Invalid interface name length\n");
        return EXIT_FAILURE;
    }
    memcpy(ifname, interface, iflen);
    ifname[iflen] = '\0';

    // Create a UDP socket with close-on-exec hardening
    sockfd = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (sockfd < 0) {
        perror("socket");
        return EXIT_FAILURE;
    }

    // Bind the socket to a specific network interface (requires CAP_NET_ADMIN)
    // Ensure optlen includes the terminating null (CWE-170)
    if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, ifname, (socklen_t)(iflen + 1)) < 0) {
        perror("setsockopt(SO_BINDTODEVICE)");
        fprintf(stderr, "Hint: This operation requires CAP_NET_ADMIN. Consider: sudo setcap 'cap_net_admin=+ep' <binary>\n");
        close(sockfd);
        return EXIT_FAILURE;
    }

    // Drop privileges as soon as the privileged operation is done (CWE-250)
    if (drop_privileges_to_real_ids() != 0) {
        fprintf(stderr, "Failed to drop privileges safely\n");
        close(sockfd);
        return EXIT_FAILURE;
    }

    // Prepare bind address
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(12345);

    // Validate IP address conversion (CWE-20)
    int rc = inet_pton(AF_INET, ip, &addr.sin_addr);
    if (rc != 1) {
        if (rc == 0) {
            fprintf(stderr, "Invalid IPv4 address: %s\n", ip);
        } else {
            perror("inet_pton");
        }
        close(sockfd);
        return EXIT_FAILURE;
    }

    // Bind the socket to the specific IP address
    if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sockfd);
        return EXIT_FAILURE;
    }

    printf("Socket successfully bound to %s on interface %s\n", ip, ifname);

    // ... use the socket ...

    // Clean shutdown (CWE-775)
    if (close(sockfd) != 0) {
        perror("close");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```

Notes
- If you need to run this without full root, grant only CAP_NET_ADMIN to the binary: sudo setcap 'cap_net_admin=+ep' ./yourprog. This limits privileges (principle of least privilege).
- If the IP may not be assigned yet to the interface and you still want to bind, consider IP_FREEBIND (hardening/config choice, not a security fix).
- Consider adding SO_REUSEADDR as appropriate for your use-case.