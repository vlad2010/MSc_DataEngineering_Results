#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define SERVER_IP "127.0.0.1" // 서버 IP 주소
#define PORT 3001
#define BUFFER_SIZE 64 // Assuming maximum data size of 64 integers

int main() {
    int client_socket;
    struct sockaddr_in server_addr;
    int buffer[BUFFER_SIZE]; // Data will be stored as integers in the buffer
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
        exit(EXIT_FAILURE);
    }

    while (1) {
        do {
            flag = 0;
            printf("Client (You): ");
            char input[256];
            if (!fgets(input, sizeof(input), stdin)) {
                printf("입력 오류가 발생하였습니다.\n");
                flag = 1;
                continue;
            }
            input[strcspn(input, "\n")] = '\0'; // Remove newline character
            char* token = strtok(input, " ");
            int index = 0;

            // Convert the space-separated input to integers and store in buffer
            while (token != NULL && index < BUFFER_SIZE) {
                buffer[index] = atoi(token);
                token = strtok(NULL, " ");
                index++;
            }

            // Check if there are too many integers to fit in the buffer
            if (token != NULL) {
                printf("Buffer size exceeded. Only the first %d integers will be sent.\n", BUFFER_SIZE);
            }

            int send_result = sendto(client_socket, buffer, sizeof(int) * index, 0, (const struct sockaddr *)&server_addr, sizeof(server_addr));
            if (send_result < 0) {
                perror("sendto failed");
                flag = 1;
            }
        } while (flag);

        // Receive data from server
        ssize_t received_bytes = recvfrom(client_socket, buffer, sizeof(buffer), 0, NULL, NULL);
