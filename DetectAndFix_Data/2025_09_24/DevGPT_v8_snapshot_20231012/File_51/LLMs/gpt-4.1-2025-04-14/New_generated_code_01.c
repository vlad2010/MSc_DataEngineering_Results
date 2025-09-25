#include <stdio.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <string.h>
#include <errno.h>

// ... (other necessary code, e.g., socket setup)

struct sockaddr_in client_addr;
socklen_t client_addr_size = sizeof(client_addr);

// Receive number from client
uint32_t receivedData = 0;
ssize_t bytes_received = recvfrom(
    server_socket,
    &receivedData,
    sizeof(receivedData),
    0,
    (struct sockaddr *)&client_addr,
    &client_addr_size
);

if (bytes_received < 0) {
    // Handle error
    perror("recvfrom failed");
    // exit or return as appropriate
} else if ((size_t)bytes_received != sizeof(receivedData)) {
    // Handle incomplete data
    fprintf(stderr, "Incomplete data received: expected %zu, got %zd\n", sizeof(receivedData), bytes_received);
    // exit or return as appropriate
} else {
    uint32_t bigEndianData = ntohl(receivedData);
    printf("Client: %x\n", bigEndianData);
}