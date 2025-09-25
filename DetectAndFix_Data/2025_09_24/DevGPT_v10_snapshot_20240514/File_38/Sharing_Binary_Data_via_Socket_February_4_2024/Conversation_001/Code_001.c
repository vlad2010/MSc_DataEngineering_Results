#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define PORT 8080

int main() {
    int sockfd;
    struct sockaddr_in server_addr;
    char *data_to_send = "Hello, this is raw binary data!";
    size_t data_len = strlen(data_to_send);

    // Create socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Set up server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    // Connect to server
    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection failed");
        exit(EXIT_FAILURE);
    }

    // Send data
    ssize_t bytes_sent = send(sockfd, data_to_send, data_len, 0);
    if (bytes_sent < 0) {
        perror("Send failed");
        exit(EXIT_FAILURE);
    }

    printf("Sent %zd bytes of raw binary data\n", bytes_sent);

    // Close socket
    close(sockfd);

    return 0;
}