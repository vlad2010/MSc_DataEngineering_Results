#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>

#define SERVER_IP "127.0.0.1"
#define PORT 3001
#define BUFFER_SIZE 258 // Max application buffer

static void flush_stdin_line(void) {
    int ch;
    while ((ch = getchar()) != '\n' && ch != EOF) { /* discard */ }
}

int main(void) {
    int client_socket = -1;
    struct sockaddr_in server_addr;
    char buffer[BUFFER_SIZE];

    // Create UDP socket
    client_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (client_socket < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Optional: set I/O timeouts to avoid hanging (CWE-400)
    struct timeval tv;
    tv.tv_sec = 5;   // 5s recv timeout
    tv.tv_usec = 0;
    if (setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        perror("setsockopt SO_RCVTIMEO failed");
        // Not fatal; continue
    }
    tv.tv_sec = 5;   // 5s send timeout
    tv.tv_usec = 0;
    if (setsockopt(client_socket, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) < 0) {
        perror("setsockopt SO_SNDTIMEO failed");
        // Not fatal; continue
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if (inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr) != 1) {
        perror("inet_pton failed");
        close(client_socket);
        exit(EXIT_FAILURE);
    }

    // Connect the UDP socket to the server (mitigates CWE-300/CWE-940)
    if (connect(client_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect (UDP) failed");
        close(client_socket);
        exit(EXIT_FAILURE);
    }

    for (;;) {
        // Read user input safely
        printf("Client (You): ");
        if (!fgets(buffer, sizeof(buffer), stdin)) {
            if (feof(stdin)) {
                printf("EOF received. Exiting.\n");
                break;
            }
            perror("Input error");
            continue;
        }

        // If the line was longer than buffer, flush the remainder (robustness)
        size_t len = strnlen(buffer, sizeof(buffer));
        if (len > 0 && buffer[len - 1] != '\n') {
            flush_stdin_line();
        }

        // Strip trailing newline if present
        buffer[strcspn(buffer, "\n")] = '\0';
        len = strnlen(buffer, sizeof(buffer)); // recompute after stripping

        // Send without the trailing NUL (avoid unnecessary byte)
        if (len > 0) {
            ssize_t sent = send(client_socket, buffer, len, 0);
            if (sent < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    fprintf(stderr, "send timeout\n");
                } else {
                    perror("send failed");
                }
                continue;
            }
            if ((size_t)sent != len) {
                fprintf(stderr, "Partial send on UDP (unexpected). Sent=%zd, Expected=%zu\n", sent, len);
                continue;
            }
        } else {
            // Allow empty message if desired; skip sending otherwise
            // continue;
        }

        // Receive data from the connected peer
        // Use recvmsg with MSG_TRUNC to detect oversized datagrams (CWE-20)
        struct iovec iov;
        struct msghdr msg;
        memset(&msg, 0, sizeof(msg));
        iov.iov_base = buffer;
        iov.iov_len = sizeof(buffer);
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;

        // On Linux, passing MSG_TRUNC causes recvmsg to return the full datagram length
        // even if it doesn't fit in the buffer, while copying only buffer-sized bytes.
        ssize_t received = recvmsg(client_socket, &msg,
#ifdef MSG_TRUNC
                                   MSG_TRUNC
#else
                                   0
#endif
        );

        if (received < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                fprintf(stderr, "recv timeout\n");
                continue;
            }
            perror("recv failed");
            continue;
        }

        // Detect truncation and drop oversized datagrams safely
        int truncated = (msg.msg_flags & MSG_TRUNC) ? 1 : 0;
#ifdef MSG_TRUNC
        // If MSG_TRUNC was in flags, 'received' is the full datagram length on Linux.
        size_t displayed = truncated ? (size_t)iov.iov_len : (size_t)received;
#else
        // Portable fallback: when truncated, 'received' is the number of copied bytes,
        // and we cannot know the original size; we still drop and warn.
        size_t displayed = (size_t)received;
#endif
        if (truncated) {
#ifdef MSG_TRUNC
            fprintf(stderr, "Warning: oversized datagram (%zd bytes). Dropping.\n", received);
#else
            fprintf(stderr, "Warning: oversized datagram detected. Dropping.\n");
#endif
            // Discard this datagram (do not process partial content)
            continue;
        }

        // Print the received data in hexadecimal format
        printf("Server: ");
        for (size_t i = 0; i < displayed; i++) {
            printf("%02x ", (unsigned char)buffer[i]);
        }
        printf("\n");
    }

    close(client_socket);
    return 0;
}