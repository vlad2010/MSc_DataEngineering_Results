#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <errno.h>
#include <stdint.h>
#include <sys/time.h>

#define SERVER_IP "169.254.14.229" // Replace with the server's IP address
#define PORT 8080

static int read_int32_from_stdin(int32_t *out) {
    char line[64];
    if (!fgets(line, sizeof(line), stdin)) {
        return 0; // EOF or error
    }

    // Trim leading spaces
    char *p = line;
    while (*p == ' ' || *p == '\t') p++;

    errno = 0;
    char *end = NULL;
    long val = strtol(p, &end, 10);

    // Ensure we parsed something
    if (p == end) {
        fprintf(stderr, "Invalid input: not a number\n");
        return -1;
    }

    // Skip trailing spaces
    while (*end == ' ' || *end == '\t') end++;

    // Allow trailing newline; otherwise, extra junk is invalid
    if (*end != '\n' && *end != '\0') {
        fprintf(stderr, "Invalid input: trailing characters\n");
        return -1;
    }

    if (errno == ERANGE || val < INT32_MIN || val > INT32_MAX) {
        fprintf(stderr, "Invalid input: out of 32-bit range\n");
        return -1;
    }

    *out = (int32_t)val;
    return 1; // success
}

int main(void) {
    int client_socket = -1;
    struct sockaddr_in server_addr;

    // Create UDP socket
    client_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (client_socket < 0) {
        perror("socket creation failed");
        return EXIT_FAILURE;
    }

    // Optional: set a receive timeout to avoid hanging forever
    struct timeval tv = { .tv_sec = 5, .tv_usec = 0 };
    if (setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        perror("setsockopt(SO_RCVTIMEO) failed");
        // Not fatal; continue
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if (inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr) != 1) {
        perror("Invalid address / Address not supported");
        close(client_socket);
        return EXIT_FAILURE;
    }

    // "Connect" the UDP socket to lock the peer endpoint and simplify recv path.
    if (connect(client_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect failed");
        close(client_socket);
        return EXIT_FAILURE;
    }

    for (;;) {
        printf("Client (You): ");
        fflush(stdout);

        int32_t number = 0;
        int r = read_int32_from_stdin(&number);
        if (r == 0) { // EOF
            fprintf(stderr, "\nEOF received, exiting.\n");
            break;
        } else if (r < 0) {
            // Invalid input; prompt again
            continue;
        }

        // Convert to network byte order and send a fixed-size payload.
        uint32_t net_num = htonl((uint32_t)number);
        ssize_t sent = send(client_socket, &net_num, sizeof(net_num), 0);
        if (sent < 0) {
            if (errno == EINTR) continue;
            perror("send failed");
            break;
        }
        if ((size_t)sent != sizeof(net_num)) {
            fprintf(stderr, "Partial send (unexpected for UDP): sent=%zd\n", sent);
            // For UDP, this is unusual; treat as error.
            break;
        }

        // Receive response; expect exactly 4 bytes (uint32_t)
        uint32_t resp_net = 0;
        ssize_t recvd = recv(client_socket, &resp_net, sizeof(resp_net), 0);
        if (recvd < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                fprintf(stderr, "Receive timed out\n");
                continue;
            }
            perror("recv failed");
            break;
        }
        if ((size_t)recvd != sizeof(resp_net)) {
            fprintf(stderr, "Unexpected datagram size: got=%zd bytes, expected=%zu\n",
                    recvd, sizeof(resp_net));
            // Discard and continue; could also implement a loop to drain.
            continue;
        }

        int32_t resp_host = (int32_t)ntohl(resp_net);
        printf("Server: %d\n", resp_host);
    }

    close(client_socket);
    return EXIT_SUCCESS;
}