#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>

// Assume server_socket, client_addr, and client_addr_size are defined and initialized properly somewhere

void receive_and_process_data(int server_socket) {
    struct sockaddr_in client_addr;
    socklen_t client_addr_size = sizeof(client_addr);

    // Receive number from client
    uint32_t receivedData;
    ssize_t receivedBytes = recvfrom(server_socket, &receivedData, sizeof(uint32_t), 0, (struct sockaddr *)&client_addr, &client_addr_size);

    // Error checking for recvfrom
    if (receivedBytes < 0) {
        perror("recvfrom failed");
        exit(EXIT_FAILURE);
    }

    // Check if we received the expected number of bytes
    if (receivedBytes != sizeof(uint32_t)) {
        fprintf(stderr, "Expected %lu bytes, but received %ld bytes.\n", sizeof(uint32_t), receivedBytes);
        exit(EXIT_FAILURE);
    }

    // Convert to host byte order
    uint32_t bigEndianData = ntohl(receivedData);

    // Print the received data in a secure manner
    printf("Client sent: %08x\n", bigEndianData);
}