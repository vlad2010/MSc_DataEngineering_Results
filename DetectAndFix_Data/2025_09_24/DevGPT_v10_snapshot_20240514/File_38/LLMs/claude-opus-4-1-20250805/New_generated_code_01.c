#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include <pwd.h>
#include <grp.h>

// Define your custom protocol header structure
struct my_protocol_header {
    uint16_t field1;
    uint32_t field2;
} __attribute__((packed)); // Ensure consistent layout across platforms

#define LISTEN_PORT 8080
#define MAX_PACKET_SIZE 4096
#define MIN_HEADER_SIZE sizeof(struct my_protocol_header)
#define UNPRIVILEGED_USER "nobody" // Change to appropriate user

// Function to drop privileges
int drop_privileges(const char *username) {
    struct passwd *pw = getpwnam(username);
    if (pw == NULL) {
        fprintf(stderr, "User %s not found\n", username);
        return -1;
    }
    
    // Drop supplementary groups
    if (setgroups(0, NULL) != 0) {
        perror("setgroups failed");
        return -1;
    }
    
    // Set GID before UID
    if (setgid(pw->pw_gid) != 0) {
        perror("setgid failed");
        return -1;
    }
    
    if (setuid(pw->pw_uid) != 0) {
        perror("setuid failed");
        return -1;
    }
    
    return 0;
}

int main() {
    int sockfd = -1;
    struct sockaddr_in server_addr;
    char packet[MAX_PACKET_SIZE]; // Maximum size of the packet
    int ret = EXIT_FAILURE;

    // Create socket (requires root/CAP_NET_RAW)
    sockfd = socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
    if (sockfd < 0) {
        perror("Socket creation failed (needs root/CAP_NET_RAW)");
        return EXIT_FAILURE;
    }

    // Initialize server address structure
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(LISTEN_PORT);

    // Bind socket to the specified address and port
    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        goto cleanup;
    }

    // Drop privileges after binding (if running as root)
    if (geteuid() == 0) {
        if (drop_privileges(UNPRIVILEGED_USER) < 0) {
            fprintf(stderr, "Failed to drop privileges\n");
            goto cleanup;
        }
        printf("Privileges dropped to user: %s\n", UNPRIVILEGED_USER);
    }

    // Receive packet
    ssize_t bytes_received = recvfrom(sockfd, packet, sizeof(packet), 0, NULL, NULL);
    if (bytes_received < 0) {
        perror("Recvfrom failed");
        goto cleanup;
    }

    // Validate minimum packet size before parsing
    if (bytes_received < MIN_HEADER_SIZE) {
        fprintf(stderr, "Received packet too small: %zd bytes (minimum: %zu)\n", 
                bytes_received, MIN_HEADER_SIZE);
        goto cleanup;
    }

    // Safe parsing of the custom header using memcpy to avoid alignment issues
    struct my_protocol_header received_header;
    memcpy(&received_header, packet, sizeof(received_header));
    
    // Convert from network byte order to host byte order
    uint16_t field1 = ntohs(received_header.field1);
    uint32_t field2 = ntohl(received_header.field2);

    // Validate header fields (add application-specific validation)
    if (field1 > 65535 || field2 > 4294967295U) {
        fprintf(stderr, "Invalid header field values\n");
        goto cleanup;
    }

    printf("Received custom header: field1=%u, field2=%u\n", field1, field2);
    
    // Calculate payload size safely
    size_t payload_size = bytes_received - MIN_HEADER_SIZE;
    if (payload_size > 0) {
        printf("Payload size: %zu bytes\n", payload_size);
        // Process payload data safely
        // char *payload = packet + MIN_HEADER_SIZE;
        // Process payload with bounds checking...
    }

    ret = EXIT_SUCCESS;

cleanup:
    // Always close socket if it was opened
    if (sockfd >= 0) {
        if (close(sockfd) < 0) {
            perror("Failed to close socket");
            ret = EXIT_FAILURE;
        }
    }

    return ret;
}