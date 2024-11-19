#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define SERVER_IP "169.254.14.229" // Replace with the server's IP address
#define PORT 8080
#define BUFFER_SIZE 1024

int main() {
    int client_socket;
    struct sockaddr_in server_addr;

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
        perror("Invalid address/ Address not supported");
        close(client_socket);
        exit(EXIT_FAILURE);
    }

    int number;
    char input[BUFFER_SIZE];
    ssize_t bytes_sent, bytes_received;

    while (1) {
        // Get input from user safely
        printf("Client (You, type 'exit' to quit): ");
        if (!fgets(input, sizeof(input), stdin)) {
            break; // EOF or read error
        }

        // Check for exit condition
        if (strncmp(input, "exit", 4) == 0) {
            break;
        }

        // Attempt to convert input to an integer
        if (sscanf(input, "%d", &number) != 1) {
            fprintf(stderr, "Invalid input. Please enter a valid integer.\n");
            continue;
        }

        // Send number to server
        bytes_sent = sendto(client_socket, &number, sizeof(int), 0,
                            (const struct sockaddr *)&server_addr, sizeof(server_addr));
        if (bytes_sent != sizeof(int)) {
            perror("Failed to send data");
            continue;
        }

        // Receive number from server
        bytes_received = recvfrom(client_socket, &number, sizeof(int), 0, NULL, NULL);
        if (bytes_received != sizeof(int)) {
            perror("Failed to receive data");
            continue;
        }
        printf("Server: %d\n", number);
    }

    close(client_socket);
    return 0;
}