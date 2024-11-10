#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <errno.h>
#include <stdbool.h>

#define PORT 3001
#define BUFFER_SIZE 64 // Max number of integers
#define INPUT_SIZE 256 // Length of input buffer

bool is_number(const char *str) {
    while (*str) {
        if (*str < '0' || *str > '9') return false;
        str++;
    }
    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <server_ip>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char *server_ip = argv[1];

    int client_socket;
    struct sockaddr_in server_addr;
    int buffer[BUFFER_SIZE];
    char input[INPUT_SIZE];
    int flag;

    // Create socket
    if ((client_socket = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        perror("inet_pton failed");
        close(client_socket);
        exit(EXIT_FAILURE);
    }

    while (1) {
        do {
            flag = 0;
            printf("Client (You): ");
            if (!fgets(input, sizeof(input), stdin)) {
                printf("Input error occurred.\n");
                flag = 1;
                continue;
            }
            input[strcspn(input, "\n")] = '\0';

            char* token = strtok(input, " ");
            int index = 0;
            memset(buffer, 0, sizeof(buffer)); // Clear buffer before use

            while (token != NULL && index < BUFFER_SIZE) {
                if (!is_number(token)) {
                    printf("Invalid input detected. Please enter integers only.\n");
                    flag = 1;
                    break;
                }
                
                errno = 0;
                long value = strtol(token, NULL, 10);
                if (errno != 0 || value > INT_MAX || value < INT_MIN) {
                    perror("Integer conversion error");
                    flag = 1;
                    break;
                }
                
                buffer[index++] = (int)value;
                token = strtok(NULL, " ");
            }

            if (token != NULL) {
                printf("Buffer size exceeded. Only the first %d integers will be sent.\n", BUFFER_SIZE);
            }

            if (flag == 0) {
                int send_result = sendto(client_socket, buffer, sizeof(int) * index, 0, 
                                         (const struct sockaddr *)&server_addr, sizeof(server_addr));
                if (send_result < 0) {
                    perror("sendto failed");
                    flag = 1;
                }
            }
        } while (flag);

        ssize_t received_bytes = recvfrom(client_socket, buffer, sizeof(buffer), 0, NULL, NULL);
        if (received_bytes < 0) {
            perror("recvfrom failed");
        } else {
            printf("Received %zd bytes from server.\n", received_bytes);
        }
    }

    close(client_socket);
    return 0;
}