#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>

#define DEFAULT_INTERFACE "eth0"
#define DEFAULT_IP "192.168.1.2"
#define DEFAULT_PORT 12345

int main(int argc, char *argv[])
{
    int sockfd = -1;
    struct sockaddr_in addr;
    int ret;
    
    // Allow configuration via command line arguments or environment variables
    const char *interface = (argc > 1) ? argv[1] : 
                           (getenv("BIND_INTERFACE") ? getenv("BIND_INTERFACE") : DEFAULT_INTERFACE);
    const char *ip_address = (argc > 2) ? argv[2] : 
                            (getenv("BIND_IP") ? getenv("BIND_IP") : DEFAULT_IP);
    int port = (argc > 3) ? atoi(argv[3]) : DEFAULT_PORT;
    
    // Validate interface name length
    if (strlen(interface) >= IFNAMSIZ) {
        fprintf(stderr, "Interface name too long\n");
        exit(EXIT_FAILURE);
    }
    
    // Create a socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }
    
    // Bind the socket to a specific network interface (requires CAP_NET_RAW or root)
    // Note: This will fail if not running with appropriate privileges
    if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, interface, strlen(interface) + 1) < 0) {
        // Log warning but continue - binding to interface is optional
        fprintf(stderr, "Warning: SO_BINDTODEVICE failed (requires CAP_NET_RAW): %s\n", strerror(errno));
        // In production, you might want to exit here if interface binding is mandatory
    }
    
    // Initialize the address structure
    memset(&addr, 0, sizeof(struct sockaddr_in));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    
    // Convert IP address with error checking
    ret = inet_pton(AF_INET, ip_address, &addr.sin_addr);
    if (ret <= 0) {
        if (ret == 0) {
            fprintf(stderr, "Invalid IP address format: %s\n", ip_address);
        } else {
            perror("inet_pton");
        }
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    // Bind the socket to the specific IP address
    if (bind(sockfd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    printf("Socket successfully bound to %s:%d on interface %s\n", 
           ip_address, port, interface);
    
    // Drop privileges if running as root (after binding)
    if (geteuid() == 0) {
        // In production, set to appropriate non-root user
        // Example: setuid(nobody_uid);
        fprintf(stderr, "Warning: Running as root. Consider dropping privileges.\n");
    }
    
    // Now you can use the socket...
    // Example: Add actual socket operations here
    
    // Properly close the socket before exiting
    if (sockfd >= 0) {
        close(sockfd);
    }
    
    return 0;
}