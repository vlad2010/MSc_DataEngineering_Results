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
    char buffer[BUFFER_SIZE]; // 데이터를 저장할 버퍼
    int flag, c;

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
        ssize_t received_bytes = recvfrom(server_socket, buffer, sizeof(buffer), 0, (struct sockaddr *)&client_addr, &client_addr_size);
        if (received_bytes < 0) {
            perror("recvfrom failed");
            continue;
        }

        // Print the received data in hexadecimal format
        printf("Client: ");
        for (ssize_t i = 0; i < received_bytes; i++) {
            printf("%02x ", (unsigned char)buffer[i]);
        }
        printf("\n");

        // Reply to client
        do {
            flag = 0;
            printf("Server (You): ");
            if (!fgets(buffer, sizeof(buffer), stdin)) {
                printf("입력 오류가 발생하였습니다.\n");
                flag = 1;
                continue;
            }
            buffer[strcspn(buffer, "\n")] = '\0'; // 줄바꿈 문자 제거
            int send_result = sendto(server_socket, buffer, strlen(buffer), 0, (const struct sockaddr *)&client_addr, client_addr_size);
            if (send_result < 0) {
                perror("sendto failed");
                flag = 1;
            }
        } while (flag);
    }

    close(server_socket);
    return 0;
}
