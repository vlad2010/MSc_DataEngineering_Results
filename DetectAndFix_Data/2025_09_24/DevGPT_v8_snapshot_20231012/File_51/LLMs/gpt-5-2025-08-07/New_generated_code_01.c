#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

// Receives exactly one uint32_t (network byte order) datagram and prints it in hex.
// Drops any datagram that is not exactly 4 bytes or that is truncated.
void receive_uint32_datagram(int server_socket) {
    struct sockaddr_storage client_addr;
    uint32_t net_u32 = 0;

    // Set up a single IO vector for receiving 4 bytes
    struct iovec iov = {
        .iov_base = &net_u32,
        .iov_len  = sizeof(net_u32)
    };

    for (;;) {
        struct msghdr msg;
        memset(&msg, 0, sizeof(msg));

        socklen_t client_addr_len = sizeof(client_addr); // Proper initialization (CWE-665)
        msg.msg_name    = &client_addr;
        msg.msg_namelen = client_addr_len;
        msg.msg_iov     = &iov;
        msg.msg_iovlen  = 1;

        ssize_t n = recvmsg(server_socket, &msg, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue; // retry on signal interruption
            }
            perror("recvmsg");
            return;
        }

        // If the message was truncated or not exactly 4 bytes, drop it
        if ((msg.msg_flags & MSG_TRUNC) || n != (ssize_t)sizeof(uint32_t)) {
            // Drop invalid/oversized datagram and continue to next
            continue;
        }

        // Safe: net_u32 fully initialized with exactly 4 bytes from the datagram
        uint32_t host_u32 = ntohl(net_u32);

        // Use correct format for uint32_t (CWE-686)
        printf("Client: 0x%" PRIx32 "\n", host_u32);
        break; // processed one valid datagram
    }
}