#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <errno.h>
#include <sys/time.h>

#define SERVER_IP "127.0.0.1"
#define PORT 3001
#define BUFFER_SIZE 258
#define TIMEOUT_SEC 5

int main() {
    int client_socket;
    struct sockaddr_in server_addr, recv_addr;
    char buffer[BUFFER_SIZE];
    char recv_buffer[BUFFER_SIZE]; // Separate buffer for receiving
    int flag;
    socklen_t addr_len;
    struct timeval timeout;

    // Create socket
    if ((client_socket = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Set socket timeout for recvfrom to prevent indefinite blocking
    timeout.tv_sec = TIMEOUT_SEC;
    timeout.tv_usec = 0;
    if (setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
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
            fflush(stdout);
            
            // Clear buffer before use
            memset(buffer, 0, sizeof(buffer));
            
            if (!fgets(buffer, sizeof(buffer) - 1, stdin)) { // Leave room for null terminator
                if (feof(stdin)) {
                    printf("\nExiting...\n");
                    close(client_socket);
                    exit(EXIT_SUCCESS);
                }
                printf("Input error occurred.\n");
                flag = 1;
                clearerr(stdin);
                continue;
            }
            
            // Remove newline character safely
            size_t len = strlen(buffer);
            if (len > 0 && buffer[len - 1] == '\n') {
                buffer[len - 1] = '\0';
                len--;
            }
            
            // Validate input length
            if (len >= BUFFER_SIZE - 1) {
                printf("Input too long. Maximum %d characters allowed.\n", BUFFER_SIZE - 2);
                flag = 1;
                continue;
            }
            
            // Send data with explicit length
            ssize_t send_result = sendto(client_socket, buffer, len + 1, 0, 
                                        (const struct sockaddr *)&server_addr, 
                                        sizeof(server_addr));
            if (send_result < 0) {
                perror("sendto failed");
                flag = 1;
            }
        } while (flag);

        // Clear receive buffer
        memset(recv_buffer, 0, sizeof(recv_buffer));
        addr_len = sizeof(recv_addr);
        memset(&recv_addr, 0, sizeof(recv_addr));
        
        // Receive data from server with sender verification
        ssize_t received_bytes = recvfrom(client_socket, recv_buffer, 
                                         sizeof(recv_buffer) - 1, 0, 
                                         (struct sockaddr *)&recv_addr, &addr_len);
        
        if (received_bytes < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                printf("Receive timeout - no response from server.\n");
            } else {
                perror("recvfrom failed");
            }
            continue;
        }
        
        // Verify the response came from the expected server
        if (recv_addr.sin_addr.s_addr != server_addr.sin_addr.s_addr ||
            recv_addr.sin_port != server_addr.sin_port) {
            char recv_ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &recv_addr.sin_addr, recv_ip, INET_ADDRSTRLEN);
            printf("Warning: Received data from unexpected source %s:%d\n", 
                   recv_ip, ntohs(recv_addr.sin_port));
            continue;
        }
        
        // Ensure null termination for safety
        recv_buffer[received_bytes] = '\0';
        
        // Validate received data size
        if (received_bytes <= 0 || received_bytes >= BUFFER_SIZE) {
            printf("Invalid response size: %zd bytes\n", received_bytes);
            continue;
        }
        
        // Print the received data in hexadecimal format
        printf("Server (%zd bytes): ", received_bytes);
        for (ssize_t i = 0; i < received_bytes; i++) {
            printf("%02x ", (unsigned char)recv_buffer[i]);
            // Add line break for readability on long outputs
            if ((i + 1) % 16 == 0 && i + 1 < received_bytes) {
                printf("\n         ");
            }
        }
        printf("\n");
    }

    close(client_socket);
    return 0;
}