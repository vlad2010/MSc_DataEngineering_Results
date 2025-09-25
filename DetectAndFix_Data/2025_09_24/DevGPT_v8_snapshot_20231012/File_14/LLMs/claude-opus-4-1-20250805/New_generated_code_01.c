#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <signal.h>
#include <errno.h>
#include <ctype.h>
#include <fcntl.h>
#include <sys/select.h>

#define PORT 3001
#define BUFFER_SIZE 258
#define MAX_MESSAGE_SIZE 256
#define TIMEOUT_SEC 5

volatile sig_atomic_t keep_running = 1;
int server_socket = -1;

void signal_handler(int sig) {
    keep_running = 0;
}

void cleanup() {
    if (server_socket >= 0) {
        if (close(server_socket) < 0) {
            perror("close failed");
        }
        server_socket = -1;
    }
}

int is_printable_data(const unsigned char *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (!isprint(data[i]) && !isspace(data[i])) {
            return 0;
        }
    }
    return 1;
}

int main() {
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_size;
    char buffer[BUFFER_SIZE];
    char input_buffer[BUFFER_SIZE];
    int flag;
    fd_set readfds;
    struct timeval tv;
    int max_fd;

    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Create socket
    if ((server_socket = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Set socket to non-blocking mode for select()
    int flags = fcntl(server_socket, F_GETFL, 0);
    if (flags < 0 || fcntl(server_socket, F_SETFL, flags | O_NONBLOCK) < 0) {
        perror("fcntl failed");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // Enable SO_REUSEADDR
    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt failed");
        cleanup();
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));

    // Configure server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    // Bind the socket to the server address
    if (bind(server_socket, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        cleanup();
        exit(EXIT_FAILURE);
    }

    printf("Server started on port %d. Waiting for messages...\n", PORT);
    printf("Press Ctrl+C to exit gracefully.\n");

    while (keep_running) {
        FD_ZERO(&readfds);
        FD_SET(server_socket, &readfds);
        FD_SET(STDIN_FILENO, &readfds);
        max_fd = (server_socket > STDIN_FILENO) ? server_socket : STDIN_FILENO;

        tv.tv_sec = 1;
        tv.tv_usec = 0;

        int activity = select(max_fd + 1, &readfds, NULL, NULL, &tv);
        
        if (activity < 0) {
            if (errno == EINTR) continue;
            perror("select error");
            break;
        }

        // Check for incoming UDP messages
        if (FD_ISSET(server_socket, &readfds)) {
            client_addr_size = sizeof(client_addr);
            memset(&client_addr, 0, sizeof(client_addr));
            memset(buffer, 0, sizeof(buffer));

            ssize_t received_bytes = recvfrom(server_socket, buffer, MAX_MESSAGE_SIZE, 0, 
                                             (struct sockaddr *)&client_addr, &client_addr_size);
            
            if (received_bytes < 0) {
                if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    perror("recvfrom failed");
                }
                continue;
            }

            // Validate received data size
            if (received_bytes > MAX_MESSAGE_SIZE) {
                printf("Warning: Received oversized message (%zd bytes), truncated\n", received_bytes);
                received_bytes = MAX_MESSAGE_SIZE;
            }

            // Null-terminate the buffer for safety
            buffer[received_bytes] = '\0';

            // Get client information
            char client_ip[INET_ADDRSTRLEN];
            if (inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip)) == NULL) {
                strncpy(client_ip, "unknown", sizeof(client_ip) - 1);
                client_ip[sizeof(client_ip) - 1] = '\0';
            }

            printf("\n[%s:%d] Received %zd bytes:\n", client_ip, ntohs(client_addr.sin_port), received_bytes);

            // Print data safely
            if (is_printable_data((unsigned char*)buffer, received_bytes)) {
                printf("Text: %.*s\n", (int)received_bytes, buffer);
            } else {
                printf("Hex: ");
                for (ssize_t i = 0; i < received_bytes && i < 32; i++) {
                    printf("%02x ", (unsigned char)buffer[i]);
                }
                if (received_bytes > 32) {
                    printf("... (%zd more bytes)", received_bytes - 32);
                }
                printf("\n");
            }

            // Send response
            printf("Enter response (or press Enter to skip): ");
            fflush(stdout);
        }

        // Check for stdin input
        if (FD_ISSET(STDIN_FILENO, &readfds)) {
            if (fgets(input_buffer, sizeof(input_buffer), stdin) != NULL) {
                // Remove newline
                size_t len = strcspn(input_buffer, "\n");
                input_buffer[len] = '\0';

                if (len > 0 && client_addr.sin_port != 0) {
                    // Validate message size
                    if (len > MAX_MESSAGE_SIZE) {
                        printf("Error: Message too long (%zu bytes). Maximum is %d bytes.\n", len, MAX_MESSAGE_SIZE);
                        continue;
                    }

                    ssize_t sent = sendto(server_socket, input_buffer, len, 0,
                                        (const struct sockaddr *)&client_addr, client_addr_size);
                    if (sent < 0) {
                        perror("sendto failed");
                    } else if (sent != (ssize_t)len) {
                        printf("Warning: Only sent %zd of %zu bytes\n", sent, len);
                    } else {
                        printf("Response sent (%zd bytes)\n", sent);
                    }
                } else if (client_addr.sin_port == 0) {
                    printf("No client to respond to yet.\n");
                }
            }
        }
    }

    printf("\nShutting down server...\n");
    cleanup();
    return 0;
}