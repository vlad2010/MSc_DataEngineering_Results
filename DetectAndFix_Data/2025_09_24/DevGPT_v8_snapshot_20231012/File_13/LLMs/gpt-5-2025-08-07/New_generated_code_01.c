#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h>

#define PORT 3001
#define RECV_BUF_SIZE 512  // larger than expected to detect oversize datagrams

static int recv_exact_u32(int sock, struct sockaddr_in *client_addr, socklen_t *client_addr_size, uint32_t *out_net_u32) {
    char buf[RECV_BUF_SIZE];
    for (;;) {
        *client_addr_size = sizeof(*client_addr); // reset before each call
        ssize_t n = recvfrom(sock, buf, sizeof(buf), 0, (struct sockaddr *)client_addr, client_addr_size);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("recvfrom failed");
            return -1;
        }
        if (n != (ssize_t)sizeof(uint32_t)) {
            char addrstr[INET_ADDRSTRLEN] = {0};
            inet_ntop(AF_INET, &client_addr->sin_addr, addrstr, sizeof(addrstr));
            fprintf(stderr, "Dropping datagram with unexpected length %zd from %s:%u\n",
                    n, addrstr, ntohs(client_addr->sin_port));
            return 1; // not fatal, but indicates message ignored
        }
        // Safe copy to avoid alignment issues
        memcpy(out_net_u32, buf, sizeof(uint32_t));
        return 0;
    }
}

static int send_exact_u32(int sock, const struct sockaddr_in *client_addr, socklen_t client_addr_size, uint32_t net_u32) {
    for (;;) {
        ssize_t n = sendto(sock, &net_u32, sizeof(net_u32), 0, (const struct sockaddr *)client_addr, client_addr_size);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("sendto failed");
            return -1;
        }
        if (n != (ssize_t)sizeof(uint32_t)) {
            fprintf(stderr, "Partial send: %zd bytes (expected %zu)\n", n, sizeof(uint32_t));
            return -1;
        }
        return 0;
    }
}

static int read_hex_u32_from_stdin(uint32_t *out_val) {
    char line[128];
    for (;;) {
        printf("Server (You): ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) {
            if (feof(stdin)) {
                fprintf(stderr, "EOF on stdin\n");
            } else {
                perror("fgets failed");
            }
            return -1;
        }

        // Trim trailing newline
        size_t len = strlen(line);
        if (len && line[len-1] == '\n') line[len-1] = '\0';

        // Parse hex (base 16); validate full consumption and range
        errno = 0;
        char *endp = NULL;
        unsigned long long v = strtoull(line, &endp, 16);
        if (errno == ERANGE || v > 0xFFFFFFFFULL) {
            fprintf(stderr, "Value out of range (expect 0x0 .. 0xFFFFFFFF)\n");
            continue;
        }
        // Skip trailing whitespace
        while (endp && *endp == ' ') endp++;
        if (endp == line || (endp && *endp != '\0')) {
            fprintf(stderr, "Invalid input. Enter a hexadecimal number (e.g., 1A2B3C4D).\n");
            continue;
        }

        *out_val = (uint32_t)v;
        return 0;
    }
}

int main(void) {
    int server_socket = -1;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_size;
    uint32_t net_number_in;

    // Create socket
    server_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (server_socket < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Optional hardening: allow quick rebinding after restart
    int yes = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) < 0) {
        perror("setsockopt(SO_REUSEADDR) failed");
        // not fatal
    }

    memset(&server_addr, 0, sizeof(server_addr));
    memset(&client_addr, 0, sizeof(client_addr));

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY; // consider htonl(INADDR_LOOPBACK) for local-only interactive use

    if (bind(server_socket, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(server_socket);
        exit(EXIT_FAILURE);
    }

    printf("Server started on port %d. Waiting for messages...\n", PORT);

    for (;;) {
        int r = recv_exact_u32(server_socket, &client_addr, &client_addr_size, &net_number_in);
        if (r < 0) {
            // fatal recv error
            break;
        }
        if (r > 0) {
            // length mismatch, drop and continue
            continue;
        }

        uint32_t host_number_in = ntohl(net_number_in);
        printf("Client: %" PRIx32 "\n", host_number_in);

        uint32_t host_number_out;
        if (read_hex_u32_from_stdin(&host_number_out) < 0) {
            fprintf(stderr, "Failed to read input; exiting.\n");
            break;
        }
        uint32_t net_number_out = htonl(host_number_out);

        if (send_exact_u32(server_socket, &client_addr, client_addr_size, net_number_out) < 0) {
            // non-fatal; continue serving
            continue;
        }
    }

    close(server_socket);
    return 0;
}