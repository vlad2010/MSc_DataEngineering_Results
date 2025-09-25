#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <errno.h>
#include <sys/time.h>

#define SERVER_IP "127.0.0.1" // 서버 IP 주소
#define PORT 3001
#define BUFFER_SIZE 258 // 최대 데이터 크기

int main() {
    int client_socket;
    struct sockaddr_in server_addr;
    char buffer[BUFFER_SIZE]; // 데이터를 저장할 버퍼
    int flag;

    // Create socket
    if ((client_socket = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Set a receive timeout (e.g., 5 seconds) to prevent blocking forever (CWE-400)
    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    if (setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        perror("setsockopt failed");
        close(client_socket);
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));

    // Configure server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if (inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr) <= 0) {
        perror("inet_pton failed");
        close(client_socket);
        exit(EXIT_FAILURE);
    }

    while (1) {
        do {
            flag = 0;
            printf("Client (You): ");
            if (!fgets(buffer, sizeof(buffer), stdin)) {
                printf("입력 오류가 발생하였습니다.\n");
                flag = 1;
                continue;
            }
            buffer[strcspn(buffer, "\n")] = '\0'; // 줄바꿈 문자 제거

            // Input validation: Check for empty input (CWE-20)
            if (strlen(buffer) == 0) {
                printf("입력이 비어 있습니다. 다시 입력하세요.\n");
                flag = 1;
                continue;
            }

            int send_result = sendto(client_socket, buffer, strlen(buffer) + 1, 0, (const struct sockaddr *)&server_addr, sizeof(server_addr));
            if (send_result < 0) {
                perror("sendto failed");
                flag = 1;
            }
        } while (flag);

        // Receive data from server
        ssize_t received_bytes = recvfrom(client_socket, buffer, sizeof(buffer) - 1, 0, NULL, NULL);
        if (received_bytes < 0) {
            if (errno == EWOULDBLOCK || errno == EAGAIN) {
                printf("서버 응답이 없습니다. 다시 시도하세요.\n");
            } else {
                perror("recvfrom failed");
            }
            continue;
        }

        // Ensure null-termination (CWE-120)
        buffer[received_bytes] = '\0';

        // Print the received data in hexadecimal format
        printf("Server: ");
        for (ssize_t i = 0; i < received_bytes; i++) {
            printf("%02x ", (unsigned char)buffer[i]);
        }
        printf("\n");
    }

    if (close(client_socket) < 0) {
        perror("close failed");
        return EXIT_FAILURE;
    }
    return 0;
}