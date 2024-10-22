#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 3001
#define BUFFER_SIZE sizeof(int)

int main() {
    int server_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_size = sizeof(client_addr);
    uint32_t number;
    int flag, c;
    ssize_t bytes_received, bytes_sent;

    // Create socket
    if ((server_socket = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    memset(&client_addr, 0, sizeof(client_addr));

    // Configure server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    // Bind the socket to the server address
    if (bind(server_socket, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(server_socket);
        exit(EXIT_FAILURE);
    }

    printf("Server started. Waiting for messages...\n");

    while (1) {
        // Receive number from client
        bytes_received = recvfrom(server_socket, &number, sizeof(uint32_t), 0, (struct sockaddr *)&client_addr, &client_addr_size);
        if (bytes_received < 0) {
            perror("recvfrom failed");
            continue;  // Skip this loop iteration and try again
        }

        if (bytes_received != sizeof(uint32_t)) {
            fprintf(stderr, "Received incorrect data size\n");
            continue;  // Skip this loop iteration and try again
        }

        uint32_t bigEndianData = ntohl(number);
        printf("Client: %x\n", bigEndianData);

        // Reply to client
        do {
            flag = 0;
            printf("Server (You): ");
            if (scanf("%8x", &number) != 1) {
                printf("Please enter a valid number.\n");
                flag = 1;
                while ((c = getchar()) != '\n' && c != EOF);
            } else {
                number = htonl(number);  // Convert to big-endian before sending
            }
        } while (flag);

        bytes_sent = sendto(server_socket, &number, sizeof(uint32_t), 0, (const struct sockaddr *)&client_addr, client_addr_size);
        if (bytes_sent < 0) {
            perror("sendto failed");
        }
    }

    close(server_socket);
    return 0;
}