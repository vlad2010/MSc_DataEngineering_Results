#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/poll.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>

#define PORT 3001
#define RX_BUFFER_SIZE 258   // Max UDP payload we will process/display
#define TX_BUFFER_SIZE 1024  // Max console line to send
#define MAX_PKTS_PER_SEC 200 // Basic anti-flood limit

static void print_addr(const struct sockaddr_in *addr) {
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &addr->sin_addr, ip, sizeof(ip));
    printf("%s:%u", ip, (unsigned)ntohs(addr->sin_port));
}

static int sockaddr_in_equal(const struct sockaddr_in *a, const struct sockaddr_in *b) {
    return a->sin_family == b->sin_family &&
           a->sin_addr.s_addr == b->sin_addr.s_addr &&
           a->sin_port == b->sin_port;
}

static int parse_allow_ip_from_env(struct in_addr *out) {
    // Optional allow-list: set ALLOW_IP=192.0.2.10 to only accept from that IPv4
    const char *env = getenv("ALLOW_IP");
    if (!env || !*env) return 0;
    if (inet_pton(AF_INET, env, out) != 1) {
        fprintf(stderr, "Invalid ALLOW_IP: %s\n", env);
        exit(EXIT_FAILURE);
    }
    return 1;
}

int main(void) {
    int server_socket;
    struct sockaddr_in server_addr;
    char rx_buf[RX_BUFFER_SIZE];
    char tx_buf[TX_BUFFER_SIZE];
    int locked_to_peer = 0; // If 1, socket is connected to a single peer

    // Optional allow-list by IPv4 address (exact match)
    struct in_addr allow_ip;
    int have_allow_ip = parse_allow_ip_from_env(&allow_ip);

    // Create socket
    if ((server_socket = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Make stdout unbuffered for prompt safety in interactive use
    setvbuf(stdout, NULL, _IONBF, 0);

    // SO_REUSEADDR for easier restarts
    int one = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) {
        perror("setsockopt(SO_REUSEADDR) failed");
        close(server_socket);
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_socket, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(server_socket);
        exit(EXIT_FAILURE);
    }

    printf("Server started on UDP port %d. Waiting for messages...\n", PORT);
    if (have_allow_ip) {
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &allow_ip, ip, sizeof(ip));
        printf("Allow-list active: only accepting from %s\n", ip);
    }
    printf("Commands: /lock (lock to last client), /unlock, /quit\n");

    // Track the last client we heard from
    struct sockaddr_in last_client;
    int have_last_client = 0;

    // Basic anti-flood counters
    time_t current_sec = time(NULL);
    unsigned pkt_count_this_sec = 0;

    // Poll both socket and stdin
    struct pollfd fds[2];
    fds[0].fd = server_socket;
    fds[0].events = POLLIN;
    fds[1].fd = STDIN_FILENO;
    fds[1].events = POLLIN;

    for (;;) {
        int ret = poll(fds, 2, -1);
        if (ret < 0) {
            if (errno == EINTR) continue;
            perror("poll failed");
            break;
        }

        // Handle UDP socket input
        if (fds[0].revents & POLLIN) {
            // Use recvmsg with MSG_TRUNC to detect oversized datagrams
            struct sockaddr_in client_addr;
            socklen_t client_addr_len = sizeof(client_addr);
            struct iovec iov;
            iov.iov_base = rx_buf;
            iov.iov_len = sizeof(rx_buf);
            struct msghdr msg;
            memset(&msg, 0, sizeof(msg));
            msg.msg_name = &client_addr;
            msg.msg_namelen = client_addr_len;
            msg.msg_iov = &iov;
            msg.msg_iovlen = 1;

            ssize_t n = recvmsg(server_socket, &msg, MSG_TRUNC);
            if (n < 0) {
                perror("recvmsg failed");
                continue;
            }
            // Update actual name length after recvmsg
            if (msg.msg_namelen > sizeof(client_addr)) {
                // Shouldn't happen for AF_INET; drop if it does.
                fprintf(stderr, "Received address larger than expected; dropping\n");
                continue;
            }

            // Rate limiting
            time_t now = time(NULL);
            if (now != current_sec) {
                current_sec = now;
                pkt_count_this_sec = 0;
            }
            if (++pkt_count_this_sec > MAX_PKTS_PER_SEC) {
                // Drop excess to mitigate floods
                continue;
            }

            // Optional allow-list by IPv4 address
            if (have_allow_ip && client_addr.sin_addr.s_addr != allow_ip.s_addr) {
                // Drop silently
                continue;
            }

            // Handle truncated datagram
            if (msg.msg_flags & MSG_TRUNC) {
                // Drop and warn; we never process partial payloads
                printf("Dropped oversized datagram from ");
                print_addr(&client_addr);
                printf(" (size > %d)\n", RX_BUFFER_SIZE);
                continue;
            }

            // Accept this client as the current peer (for replies) if not locked
            if (!locked_to_peer) {
                if (!have_last_client || !sockaddr_in_equal(&last_client, &client_addr)) {
                    printf("New client ");
                    print_addr(&client_addr);
                    printf(" selected for replies (not locked)\n");
                }
                last_client = client_addr;
                have_last_client = 1;
            } else {
                // If locked, ensure we only accept from the locked peer
                if (!sockaddr_in_equal(&last_client, &client_addr)) {
                    // Ignore packets not from the locked peer
                    continue;
                }
            }

            // Print the received data in hexadecimal format
            printf("Client ");
            print_addr(&client_addr);
            printf(": ");
            for (ssize_t i = 0; i < n; i++) {
                printf("%02x ", (unsigned char)rx_buf[i]);
            }
            printf("\n");
        }

        // Handle console input
        if (fds[1].revents & POLLIN) {
            if (!fgets(tx_buf, sizeof(tx_buf), stdin)) {
                if (feof(stdin)) {
                    printf("EOF on stdin. Exiting.\n");
                    break;
                }
                perror("fgets failed");
                continue;
            }
            // Remove trailing newline
            tx_buf[strcspn(tx_buf, "\n")] = '\0';

            // Commands
            if (strcmp(tx_buf, "/quit") == 0) {
                printf("Quitting.\n");
                break;
            } else if (strcmp(tx_buf, "/lock") == 0) {
                if (!have_last_client) {
                    printf("No client to lock to yet.\n");
                    continue;
                }
                if (!locked_to_peer) {
                    if (connect(server_socket, (struct sockaddr *)&last_client, sizeof(last_client)) < 0) {
                        perror("connect (lock to peer) failed");
                    } else {
                        locked_to_peer = 1;
                        printf("Locked to client ");
                        print_addr(&last_client);
                        printf(". Only packets to/from this peer will be processed.\n");
                    }
                } else {
                    printf("Already locked.\n");
                }
                continue;
            } else if (strcmp(tx_buf, "/unlock") == 0) {
                if (locked_to_peer) {
                    // Disconnect UDP socket: connect with AF_UNSPEC
                    struct sockaddr sa;
                    memset(&sa, 0, sizeof(sa));
                    sa.sa_family = AF_UNSPEC;
                    if (connect(server_socket, &sa, sizeof(sa)) < 0) {
                        perror("unlock (AF_UNSPEC connect) failed");
                    } else {
                        locked_to_peer = 0;
                        printf("Unlocked from client. Will accept any client (subject to ALLOW_IP).\n");
                    }
                } else {
                    printf("Not locked.\n");
                }
                continue;
            }

            // Send to client
            if (!have_last_client) {
                printf("No client selected yet. Wait for a client message first.\n");
                continue;
            }

            ssize_t to_send = (ssize_t)strlen(tx_buf);
            if (to_send == 0) continue;

            int send_result;
            if (locked_to_peer) {
                send_result = (int)send(server_socket, tx_buf, (size_t)to_send, 0);
            } else {
                send_result = (int)sendto(server_socket, tx_buf, (size_t)to_send, 0,
                                          (const struct sockaddr *)&last_client,
                                          (socklen_t)sizeof(last_client));
            }
            if (send_result < 0) {
                perror("send/sendto failed");
            } else {
                printf("Sent %d bytes to ", send_result);
                print_addr(&last_client);
                printf("\n");
            }
        }
    }

    close(server_socket);
    return 0;
}