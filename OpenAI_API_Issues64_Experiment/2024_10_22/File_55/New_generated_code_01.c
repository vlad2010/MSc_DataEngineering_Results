#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  // For close() function
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

int main()
{
    int sockfd;
    struct sockaddr_in addr;
    const char *preferred_interface = "eth0";
    const char *ip_address = "192.168.1.2";

    // Create a socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    // Bind the socket to a specific network interface
    if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, preferred_interface, strlen(preferred_interface)) < 0) {
        perror("SO_BINDTODEVICE");
        close(sockfd); // Close socket on error
        exit(EXIT_FAILURE);
    }

    // Check and bind the socket to a specific IP address
    memset(&addr, 0, sizeof(struct sockaddr_in));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(12345);  // Port number

    if (inet_pton(AF_INET, ip_address, &addr.sin_addr) <= 0) {
        perror("inet_pton");
        close(sockfd); // Close socket on error
        exit(EXIT_FAILURE);
    }

    if (bind(sockfd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sockfd); // Close socket on error
        exit(EXIT_FAILURE);
    }

    printf("Socket successfully bound to %s on interface %s\n", ip_address, preferred_interface);

    // Now you can use the socket...

    close(sockfd); // Always close socket before program exit
    return 0;
}