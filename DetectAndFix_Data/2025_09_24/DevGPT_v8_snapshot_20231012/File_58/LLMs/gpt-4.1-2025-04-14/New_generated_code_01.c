#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>

#define SERVER_IP "127.0.0.1"
#define PORT 3001
#define BUFFER_SIZE 64 // Maximum data size of 64 integers

int safe_atoi(const char *str, int *out) {
    char *endptr;
    long val = strtol(str, &endptr, 10);
    if (endptr == str || *endptr != '\0') {
        return 0; // Not a valid integer
    }
    if (val < INT_MIN || val > INT_MAX) {
        return 0; // Out of int range
    }
    *out = (int)val;
    return 1;
}

int main() {
    int client_socket;
    struct sockaddr_in server_addr;
    int buffer[BUFFER_SIZE];
    int flag;

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
        exit(EXIT_FAILURE);
    }

    while (1) {
        do {
            flag = 0;
            printf("Client (You): ");
            char input[256];
            if (!fgets(input, sizeof(input), stdin)) {
                printf("Input error occurred.\n");
                flag = 1;
                continue;
            }
            input[strcspn(input, "\n")] = '\0'; // Remove newline character

            char* token = strtok(input, " ");
            int index = 0;
            int valid = 1;

            // Convert the space-separated input to integers and store in buffer
            while (token != NULL && index < BUFFER_SIZE) {
                int value;
                if (!safe_atoi(token, &value)) {
                    printf("Invalid integer input: '%s'. Please enter only valid integers.\n", token);
                    valid = 0;
                    break;
                }
                buffer[index++] = value;
                token = strtok(NULL, " ");
            }

            // If invalid input, prompt again
            if (!valid) {
                flag = 1;
                continue;
            }

            // Check if there are too many integers to fit in the buffer
            if (token != NULL) {
                printf("Buffer size exceeded. Only the first %d integers will be sent.\n", BUFFER_SIZE);
                // Discard the rest of the line
                int ch;
                while ((ch = getchar()) != '\n' && ch != EOF);
            }

            int send_result = sendto(client_socket, buffer, sizeof(int) * index, 0, (const struct sockaddr *)&server_addr, sizeof(server_addr));
            if (send_result < 0) {
                perror("sendto failed");
                flag = 1;
            }
        } while (flag);

        // Receive data from server
        ssize_t received_bytes = recvfrom(client_socket, buffer, sizeof(buffer), 0, NULL, NULL);
        if (received_bytes < 0) {
            perror("recvfrom failed");
            continue;
        }
        if (received_bytes % sizeof(int) != 0) {
            printf("Warning: Received data size is not a multiple of int size.\n");
            continue;
        }
        int num_ints = received_bytes / sizeof(int);
        printf("Received %d integers from server:\n", num_ints);
        for (int i = 0; i < num_ints && i < BUFFER_SIZE; i++) {
            printf("%d ", buffer[i]);
        }
        printf("\n");
    }

    close(client_socket);
    return 0;
}