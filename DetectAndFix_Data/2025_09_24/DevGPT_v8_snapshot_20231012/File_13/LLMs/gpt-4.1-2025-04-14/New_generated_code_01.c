#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <errno.h>

#define PORT 3001
#define BUFFER_SIZE sizeof(uint32_t)

int main() {
    int server_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_size = sizeof(client_addr);
    uint32_t number;
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
        close(server_socket);
        exit(EXIT_FAILURE);
    }

    printf("Server started. Waiting for messages...\n");

    while (1) {
        ssize_t recv_len = recvfrom(server_socket, &number, BUFFER_SIZE, 0, (struct sockaddr *)&client_addr, &client_addr_size);
        if (recv_len < 0) {
            perror("recvfrom failed");
            continue; // Don't exit, but skip this iteration
        }
        if (recv_len != BUFFER_SIZE) {
            fprintf(stderr, "Received unexpected data size: %zd bytes\n", recv_len);
            continue;
        }

        uint32_t bigEndianData = ntohl(number);
        printf("Client: %x\n", bigEndianData);

        // Reply to client
        do {
            flag = 0;
            printf("Server (You): ");
            fflush(stdout); // Ensure prompt is shown

            int scanf_result = scanf("%x", &number);
            if (scanf_result != 1) {
                printf("Please enter a valid hexadecimal number.\n");
                flag = 1;
                while ((c = getchar()) != '\n' && c != EOF);
                continue;
            }
            // Optionally, validate the range of number here if needed

            number = htonl(number); // Convert to network byte order before sending
        } while (flag);

        ssize_t sent_len = sendto(server_socket, &number, BUFFER_SIZE, 0, (const struct sockaddr *)&client_addr, client_addr_size);
        if (sent_len < 0) {
            perror("sendto failed");
            // Optionally, handle error or break/continue
        }
    }

    close(server_socket);
    return 0;
}