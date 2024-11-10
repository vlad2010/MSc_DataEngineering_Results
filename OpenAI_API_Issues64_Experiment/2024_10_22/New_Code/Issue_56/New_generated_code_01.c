#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

int main()
{
    int sockfd;
    struct sockaddr_in addr;

    // Fetch IP address and port from environment variables or use defaults
    const char *ip_address = getenv("BIND_IP");
    if (ip_address == NULL) {
        ip_address = "192.168.1.2";  // Default IP, consider making it configurable
    }
    
    const char *port_str = getenv("BIND_PORT");
    int port = (port_str != NULL) ? atoi(port_str) : 12345;  // Default port, validate input

    // Create a socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return EXIT_FAILURE;  // Exit gracefully
    }

    // Enable SO_REUSEADDR
    int enable = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
        perror("SO_REUSEADDR");
        close(sockfd);
        return EXIT_FAILURE;  // Exit gracefully
    }

    // Bind the socket to a specific IP address and port
    memset(&addr, 0, sizeof(struct sockaddr_in));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip_address, &addr.sin_addr) <= 0) {
        perror("inet_pton");
        close(sockfd);
        return EXIT_FAILURE;  // Exit gracefully
    }

    if (bind(sockfd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sockfd);
        return EXIT_FAILURE;  // Exit gracefully
    }

    printf("Socket successfully bound to %s:%d with SO_REUSEADDR enabled\n", ip_address, port);

    // Clean up when done
    close(sockfd);
    return EXIT_SUCCESS;
}