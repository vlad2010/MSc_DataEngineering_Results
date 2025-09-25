#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdint.h>
#include <errno.h>

// Define your custom protocol header structure
struct my_protocol_header {
    uint16_t field1;
    uint32_t field2;
};

#define LISTEN_PORT 8080

int main() {
    int sockfd;
    struct sockaddr_in server_addr;
    char packet[4096]; // Maximum size of the packet

    // Create socket (preferably use SOCK_DGRAM for safety, unless raw is required)
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Initialize server address structure
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(LISTEN_PORT);

    // Bind socket to the specified address and port
    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    // Receive packet
    ssize_t bytes_received = recvfrom(sockfd, packet, sizeof(packet), 0, NULL, NULL);
    if (bytes_received < 0) {
        perror("Recvfrom failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    // Check that the received packet is large enough for the header
    if ((size_t)bytes_received < sizeof(struct my_protocol_header)) {
        fprintf(stderr, "Received packet too small (%zd bytes)\n", bytes_received);
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    // Parse the custom header safely
    struct my_protocol_header received_header;
    memcpy(&received_header, packet, sizeof(received_header));
    uint16_t field1 = ntohs(received_header.field1);
    uint32_t field2 = ntohl(received_header.field2);

    // Validate header fields (example: ensure field1 and field2 are within expected ranges)
    if (field1 > 1000) { // Example validation
        fprintf(stderr, "Invalid field1 value: %u\n", field1);
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    printf("Received custom header: field1=%u, field2=%u\n", field1, field2);

    // Process payload data (if any)
    // You can access the payload data beyond the custom header in the packet buffer

    // Close socket
    close(sockfd);

    return 0;
}