#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define SERVER_IP "127.0.0.1" // Server IP address
#define PORT 3001
#define BUFFER_SIZE 258 // Maximum data size

int main() {
    int client_socket;
    struct sockaddr_in server_addr;
    char buffer[BUFFER_SIZE]; // Buffer to store data
    int flag, c;

    // Create socket
    if ((client_socket = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));

    // Configure server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if (inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr) <= 0) {
        perror("inet_pton failed");
        close(client_socket); // Ensuring resources are released
        exit(EXIT_FAILURE);
    }

    while (1) {
        do {
            flag = 0;
            printf("Client (You): ");
            if (!fgets(buffer, sizeof(buffer), stdin)) {
                printf("Input error occurred.\n");
                flag = 1;
                continue;
            }
            buffer[strcspn(buffer, "\n")] = '\0'; // Remove newline character

            size_t buffer_length = strlen(buffer) + 1;
            if (buffer_length > sizeof(buffer)) {
                fprintf(stderr, "Input is too long.\n");
                flag = 1;
                continue;
            }

            int send_result = sendto(client_socket, buffer, buffer_length, 0, (const struct sockaddr *)&server_addr, sizeof(server_addr));
            if (send_result < 0) {
                perror("sendto failed");
                flag = 1;
            }
        } while (flag);

        // Receive data from server
        ssize_t received_bytes = recvfrom(client_socket, buffer, sizeof(buffer) - 1, 0, NULL, NULL);
        if (received_bytes < 0) {
            perror("recvfrom failed");
            continue;
        }

        buffer[received_bytes] = '\0'; // Null-terminate the received buffer

        // Print the received data in hexadecimal format
        printf("Server: ");
        for (ssize_t i = 0; i < received_bytes; i++) {
            printf("%02x ", (unsigned char)buffer[i]);
        }
        printf("\n");
    }

    close(client_socket);
    return 0;
}