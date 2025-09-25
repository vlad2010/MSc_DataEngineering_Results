#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>

#define SERVER_IP "169.254.14.229" // Replace with the server's IP address
#define PORT 8080
#define BUFFER_SIZE sizeof(int)

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
    char input_buf[64];

    while (1) {
        // Secure input reading and validation
        printf("Client (You): ");
        if (!fgets(input_buf, sizeof(input_buf), stdin)) {
            fprintf(stderr, "Input error\n");
            break;
        }
        char *endptr;
        errno = 0;
        long val = strtol(input_buf, &endptr, 10);
        if (errno != 0 || endptr == input_buf || *endptr != '\n' || val < INT_MIN || val > INT_MAX) {
            fprintf(stderr, "Invalid input. Please enter a valid integer.\n");
            continue;
        }
        number = (int)val;

        // Convert to network byte order
        int net_number = htonl(number);

        ssize_t sent = sendto(client_socket, &net_number, sizeof(net_number), 0,
                              (const struct sockaddr *)&server_addr, sizeof(server_addr));
        if (sent != sizeof(net_number)) {
            perror("sendto failed");
            break;
        }

        // Receive number from server
        int recv_number;
        ssize_t recvd = recvfrom(client_socket, &recv_number, sizeof(recv_number), 0, NULL, NULL);
        if (recvd != sizeof(recv_number)) {
            perror("recvfrom failed");
            break;
        }

        // Convert from network byte order
        number = ntohl(recv_number);
        printf("Server: %d\n", number);
    }

    close(client_socket);
    return 0;
}