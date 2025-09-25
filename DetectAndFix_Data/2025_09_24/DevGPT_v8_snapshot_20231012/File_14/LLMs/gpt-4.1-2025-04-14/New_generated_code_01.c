#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 3001
#define BUFFER_SIZE 258 // 최대 데이터 크기

int main() {
    int server_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_size = sizeof(client_addr);
    char recv_buffer[BUFFER_SIZE]; // 데이터를 저장할 버퍼 (network)
    char input_buffer[BUFFER_SIZE]; // 사용자 입력 버퍼
    int flag;

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
        exit(EXIT_FAILURE);
    }

    printf("Server started. Waiting for messages...\n");

    while (1) {
        // Receive data from client
        ssize_t received_bytes = recvfrom(server_socket, recv_buffer, sizeof(recv_buffer), 0, (struct sockaddr *)&client_addr, &client_addr_size);
        if (received_bytes < 0) {
            perror("recvfrom failed");
            continue;
        }

        // Always null-terminate the buffer for safety if using as string
        if (received_bytes >= BUFFER_SIZE) {
            // Should never happen, but just in case
            received_bytes = BUFFER_SIZE - 1;
        }
        recv_buffer[received_bytes] = '\0';

        // Print the received data in hexadecimal format
        printf("Client: ");
        for (ssize_t i = 0; i < received_bytes; i++) {
            printf("%02x ", (unsigned char)recv_buffer[i]);
        }
        printf("\n");

        // Reply to client
        do {
            flag = 0;
            printf("Server (You): ");
            if (!fgets(input_buffer, sizeof(input_buffer), stdin)) {
                printf("입력 오류가 발생하였습니다.\n");
                flag = 1;
                continue;
            }
            // Check if input was truncated
            if (strchr(input_buffer, '\n') == NULL) {
                // Input too long, flush stdin
                int ch;
                while ((ch = getchar()) != '\n' && ch != EOF);
                printf("입력이 너무 깁니다. 최대 %d자까지 입력 가능합니다.\n", BUFFER_SIZE - 2);
                flag = 1;
                continue;
            }
            input_buffer[strcspn(input_buffer, "\n")] = '\0'; // 줄바꿈 문자 제거
            int send_result = sendto(server_socket, input_buffer, strlen(input_buffer), 0, (const struct sockaddr *)&client_addr, client_addr_size);
            if (send_result < 0) {
                perror("sendto failed");
                flag = 1;
            }
        } while (flag);
    }

    close(server_socket);
    return 0;
}