#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <errno.h>
#include <stdint.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/socket.h>

#define SERVER_IP "127.0.0.1"
#define PORT 3001
#define BUFFER_SIZE 64  // Number of 32-bit integers max per message

static int parse_int32(const char *s, int32_t *out) {
    if (!s || !*s) return 0;
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno == ERANGE || end == s || *end != '\0') return 0; // invalid or overflow
    if (v < INT32_MIN || v > INT32_MAX) return 0;
    *out = (int32_t)v;
    return 1;
}

int main(void) {
    int client_socket = -1;
    struct sockaddr_in server_addr;

    // Create UDP socket
    client_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (client_socket < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Set receive timeout (avoid indefinite blocking)
    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    if (setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        perror("setsockopt(SO_RCVTIMEO) failed");
        close(client_socket);
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if (inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr) <= 0) {
        perror("inet_pton failed");
        close(client_socket);
        exit(EXIT_FAILURE);
    }

    // Connect the UDP socket to the server to enforce peer validation
    if (connect(client_socket, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect failed");
        close(client_socket);
        exit(EXIT_FAILURE);
    }

    for (;;) {
        printf("Client (You) [type 'quit' to exit]: ");
        char input[256];
        if (!fgets(input, sizeof(input), stdin)) {
            fprintf(stderr, "입력 오류가 발생하였습니다.\n");
            continue;
        }

        // Strip trailing newline
        input[strcspn(input, "\n")] = '\0';
        if (strcmp(input, "quit") == 0) {
            break;
        }

        // Tokenize and parse integers safely
        uint32_t send_buf[BUFFER_SIZE];
        size_t count = 0;

        char *saveptr = NULL;
        char *token = strtok_r(input, " ", &saveptr);
        int had_error = 0;

        while (token && count < BUFFER_SIZE) {
            int32_t val;
            if (!parse_int32(token, &val)) {
                fprintf(stderr, "Invalid integer token: '%s'\n", token);
                had_error = 1;
                break;
            }
            // Convert to network byte order (uint32_t to be well-defined)
            send_buf[count++] = htonl((uint32_t)val);
            token = strtok_r(NULL, " ", &saveptr);
        }

        if (had_error) {
            // Skip sending on parse error
            continue;
        }

        if (token != NULL) {
            fprintf(stderr, "Buffer size exceeded. Only the first %d integers will be sent.\n", BUFFER_SIZE);
        }

        if (count == 0) {
            fprintf(stderr, "No valid integers to send.\n");
            continue;
        }

        ssize_t expected_bytes = (ssize_t)(count * sizeof(uint32_t));
        ssize_t sent = send(client_socket, send_buf, (size_t)expected_bytes, 0);
        if (sent < 0) {
            perror("send failed");
            continue;
        }
        if (sent != expected_bytes) {
            fprintf(stderr, "Partial send: expected %zd bytes, sent %zd bytes\n", expected_bytes, sent);
            continue;
        }

        // Receive response (only from connected peer due to connect())
        uint32_t recv_buf[BUFFER_SIZE];
        ssize_t received_bytes = recv(client_socket, recv_buf, sizeof(recv_buf), 0);
        if (received_bytes < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                fprintf(stderr, "Receive timed out; no response from server.\n");
            } else {
                perror("recv failed");
            }
            continue;
        }

        if (received_bytes == 0) {
            fprintf(stderr, "Server closed or sent empty datagram.\n");
            continue;
        }

        if (received_bytes % (ssize_t)sizeof(uint32_t) != 0) {
            fprintf(stderr, "Warning: received size %zd is not a multiple of 4; truncating.\n", received_bytes);
        }

        size_t received_count = (size_t)(received_bytes / sizeof(uint32_t));
        printf("Server replied with %zu integers: ", received_count);
        for (size_t i = 0; i < received_count; i++) {
            int32_t host_val = (int32_t)ntohl(recv_buf[i]);
            printf("%d%s", host_val, (i + 1 < received_count) ? " " : "");
        }
        printf("\n");
    }

    close(client_socket);
    return 0;
}