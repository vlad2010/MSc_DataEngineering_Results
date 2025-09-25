#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h> // For close()

int main(int argc, char *argv[])
{
    int sockfd;
    struct sockaddr_in addr;
    const char *ip_str;
    int port;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <IP_ADDRESS> <PORT>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    ip_str = argv[1];
    port = atoi(argv[2]);
    if (port <= 0 || port > 65535) {
        fprintf(stderr, "Invalid port number: %s\n", argv[2]);
        exit(EXIT_FAILURE);
    }

    // Create a socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    // Enable SO_REUSEADDR
    int enable = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
        perror("SO_REUSEADDR");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    // Bind the socket to a specific IP address and port
    memset(&addr, 0, sizeof(struct sockaddr_in));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    int ret = inet_pton(AF_INET, ip_str, &addr.sin_addr);
    if (ret <= 0) {
        if (ret == 0) {
            fprintf(stderr, "Invalid IP address format: %s\n", ip_str);
        } else {
            perror("inet_pton");
        }
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    if (bind(sockfd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    printf("Socket successfully bound to %s:%d with SO_REUSEADDR enabled\n", ip_str, port);

    // Now you can use the socket...

    close(sockfd);
    return 0;
}