#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>

#define DEFAULT_PORT 12345
#define DEFAULT_IP "0.0.0.0"  // Bind to all interfaces by default
#define MAX_IP_LEN 46  // Support IPv6 in future

// Global socket for signal handler
volatile int g_sockfd = -1;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    if (g_sockfd >= 0) {
        close(g_sockfd);
        g_sockfd = -1;
    }
    exit(EXIT_SUCCESS);
}

// Secure socket creation with proper error handling
int create_secure_socket(const char *ip_addr, int port, int enable_reuse) {
    int sockfd;
    struct sockaddr_in addr;
    
    // Validate port range
    if (port < 1 || port > 65535) {
        fprintf(stderr, "Invalid port number: %d\n", port);
        return -1;
    }
    
    // Create a socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return -1;
    }
    
    // Set socket options based on security requirements
    if (enable_reuse) {
        // Only enable SO_REUSEADDR in development/testing environments
        // Log this as a security warning
        fprintf(stderr, "WARNING: SO_REUSEADDR enabled - not recommended for production\n");
        
        int enable = 1;
        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
            perror("SO_REUSEADDR");
            close(sockfd);
            return -1;
        }
    }
    
    // Set SO_REUSEPORT for better load balancing (Linux 3.9+)
    // This is more secure than SO_REUSEADDR
    #ifdef SO_REUSEPORT
    int reuseport = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &reuseport, sizeof(int)) < 0) {
        // Non-fatal: SO_REUSEPORT might not be available
        fprintf(stderr, "Note: SO_REUSEPORT not available\n");
    }
    #endif
    
    // Initialize address structure
    memset(&addr, 0, sizeof(struct sockaddr_in));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    
    // Convert IP address with proper error checking
    int ret = inet_pton(AF_INET, ip_addr, &addr.sin_addr);
    if (ret == 0) {
        fprintf(stderr, "Invalid IP address format: %s\n", ip_addr);
        close(sockfd);
        return -1;
    } else if (ret < 0) {
        perror("inet_pton");
        close(sockfd);
        return -1;
    }
    
    // Bind the socket
    if (bind(sockfd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sockfd);
        return -1;
    }
    
    printf("Socket successfully bound to %s:%d\n", ip_addr, port);
    
    return sockfd;
}

int main(int argc, char *argv[]) {
    char ip_addr[MAX_IP_LEN] = DEFAULT_IP;
    int port = DEFAULT_PORT;
    int enable_reuse = 0;  // Disabled by default for security
    
    // Parse command line arguments (basic example)
    if (argc > 1) {
        strncpy(ip_addr, argv[1], MAX_IP_LEN - 1);
        ip_addr[MAX_IP_LEN - 1] = '\0';  // Ensure null termination
    }
    if (argc > 2) {
        char *endptr;
        long tmp_port = strtol(argv[2], &endptr, 10);
        if (*endptr != '\0' || tmp_port < 1 || tmp_port > 65535) {
            fprintf(stderr, "Invalid port number: %s\n", argv[2]);
            exit(EXIT_FAILURE);
        }
        port = (int)tmp_port;
    }
    if (argc > 3 && strcmp(argv[3], "--enable-reuse") == 0) {
        enable_reuse = 1;
    }
    
    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Create and bind socket with security considerations
    g_sockfd = create_secure_socket(ip_addr, port, enable_reuse);
    if (g_sockfd < 0) {
        exit(EXIT_FAILURE);
    }
    
    // Example: Simple receive loop
    char buffer[1024];
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    printf("Listening for UDP packets... (Press Ctrl+C to exit)\n");
    
    while (1) {
        ssize_t n = recvfrom(g_sockfd, buffer, sizeof(buffer) - 1, 0,
                            (struct sockaddr *)&client_addr, &client_len);
        if (n < 0) {
            if (errno == EINTR) {
                continue;  // Interrupted by signal, retry
            }
            perror("recvfrom");
            break;
        }
        
        buffer[n] = '\0';  // Null terminate received data
        
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        
        printf("Received %zd bytes from %s:%d\n", 
               n, client_ip, ntohs(client_addr.sin_port));
    }
    
    // Clean up
    if (g_sockfd >= 0) {
        close(g_sockfd);
    }
    
    return 0;
}