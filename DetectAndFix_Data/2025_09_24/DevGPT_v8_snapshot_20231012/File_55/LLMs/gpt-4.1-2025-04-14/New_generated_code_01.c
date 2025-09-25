#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/if.h>      // For IFNAMSIZ
#include <unistd.h>      // For close()

int main(int argc, char *argv[])
{
    int sockfd;
    struct sockaddr_in addr;
    char interface[IFNAMSIZ];
    char ip_addr[INET_ADDRSTRLEN];

    // Accept interface and IP address as command-line arguments
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <interface> <ip_address>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Validate and copy interface name
    if (strlen(argv[1]) >= IFNAMSIZ) {
        fprintf(stderr, "Interface name too long (max %d chars)\n", IFNAMSIZ - 1);
        exit(EXIT_FAILURE);
    }
    strncpy(interface, argv[1], IFNAMSIZ - 1);
    interface[IFNAMSIZ - 1] = '\0';

    // Validate and copy IP address
    if (strlen(argv[2]) >= INET_ADDRSTRLEN) {
        fprintf(stderr, "IP address too long\n");
        exit(EXIT_FAILURE);
    }
    strncpy(ip_addr, argv[2], INET_ADDRSTRLEN - 1);
    ip_addr[INET_ADDRSTRLEN - 1] = '\0';

    // Create a socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    // Bind the socket to a specific network interface
    if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, interface, strlen(interface)) < 0) {
        perror("SO_BINDTODEVICE (requires root privileges)");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    // Bind the socket to a specific IP address
    memset(&addr, 0, sizeof(struct sockaddr_in));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(12345);  // Port number
    if (inet_pton(AF_INET, ip_addr, &addr.sin_addr) != 1) {
        fprintf(stderr, "Invalid IP address: %s\n", ip_addr);
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    if (bind(sockfd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    printf("Socket successfully bound to %s on interface %s\n", ip_addr, interface);

    // Now you can use the socket...

    close(sockfd);
    return 0;
}