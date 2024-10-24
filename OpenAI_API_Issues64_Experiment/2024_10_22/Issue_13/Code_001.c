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
    uint32_t number; // 변경된 데이터 타입 (unsigned 32비트 정수)
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
        // Receive number from client
        recvfrom(server_socket, &number, sizeof(uint32_t), 0, (struct sockaddr *)&client_addr, &client_addr_size);
        uint32_t bigEndianData = ntohl(number);
        printf("Client: %x\n", bigEndianData);

        // Reply to client
        do {
            flag = 0;
            printf("Server (You): ");
            if (!scanf("%x", &number)) {
                printf("숫자를 입력하세요.");
                flag = 1;
                while ((c = getchar()) != '\n' && c != EOF);
            }
            number = htonl(number); // 클라이언트에게 보내기 전에 빅 엔디안으로 변환
        } while (flag);

        sendto(server_socket, &number, sizeof(uint32_t), 0, (const struct sockaddr *)&client_addr, client_addr_size);
    }

    close(server_socket);
    return 0;
}
