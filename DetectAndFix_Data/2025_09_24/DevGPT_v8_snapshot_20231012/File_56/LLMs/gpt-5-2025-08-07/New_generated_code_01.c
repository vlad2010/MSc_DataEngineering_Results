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