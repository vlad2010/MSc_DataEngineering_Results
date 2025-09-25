#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <fcntl.h>

/*
  This file shows two safe receive patterns:

  1) Binary-safe receive (no implicit string use)
  2) Text-safe receive (ensures NUL termination)

  Fixes applied:
  - Use ssize_t for recv return value (CWE-681)
  - Handle EINTR/EAGAIN/EWOULDBLOCK properly (CWE-703/CWE-754)
  - Ensure NUL termination for text use (CWE-170/CWE-125)
  - Avoid assuming a full message is received in one call; caller processes using returned length
*/

static ssize_t recv_once_binary(int fd, void *buf, size_t cap) {
    // Single recv attempt that handles EINTR and returns:
    //  >0: bytes read
    //   0: peer closed
    //  -1: fatal error (errno set)
    //  -2: would block (non-blocking or timeout)
    for (;;) {
        ssize_t r = recv(fd, buf, cap, 0);
        if (r >= 0) {
            return r; // 0 means peer closed, >0 is data
        }
        if (errno == EINTR) {
            continue; // retry on signal
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return -2; // non-blocking or timed out
        }
        return -1; // fatal error
    }
}

static ssize_t recv_once_text(int fd, char *buf, size_t cap) {
    // Reads up to cap-1 bytes and ensures NUL termination for text processing.
    // Returns same semantics as recv_once_binary.
    if (cap == 0) {
        errno = EINVAL;
        return -1;
    }
    for (;;) {
        ssize_t r = recv(fd, buf, cap - 1, 0);
        if (r >= 0) {
            buf[r] = '\0'; // always terminate
            return r;      // 0 means peer closed
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // Ensure buffer is at least an empty string for callers expecting text
            if (cap > 0) buf[0] = '\0';
            return -2;
        }
        // fatal
        if (cap > 0) buf[0] = '\0';
        return -1;
    }
}

static void handle_socket_binary(int socket_fd) {
    unsigned char buffer[1024];

    ssize_t bytes_received = recv_once_binary(socket_fd, buffer, sizeof(buffer));
    if (bytes_received > 0) {
        // Process exactly bytes_received bytes from buffer safely as binary
        // Example: write to a file descriptor or parse with explicit length
        // write(STDOUT_FILENO, buffer, (size_t)bytes_received);
    } else if (bytes_received == 0) {
        // Peer performed an orderly shutdown
        // Close or cleanup connection
    } else if (bytes_received == -2) {
        // Would block / no data available right now (non-blocking or timeout)
        // Handle according to your event loop (e.g., retry later)
    } else {
        // Fatal error
        // Log errno safely and handle
        // perror("recv");
    }
}

static void handle_socket_text(int socket_fd) {
    char buffer[1024];

    // Optional: set a receive timeout to prevent indefinite blocking (helps DoS mitigation)
    // struct timeval tv = { .tv_sec = 10, .tv_usec = 0 };
    // setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    ssize_t bytes_received = recv_once_text(socket_fd, buffer, sizeof(buffer));
    if (bytes_received > 0) {
        // buffer is guaranteed NUL-terminated
        // NOTE: bytes_received is the number of bytes actually received, not including the NUL terminator.
        // Safe use as string:
        // printf("Received: %s\n", buffer);

        // If protocol messages can exceed buffer size, bytes_received == sizeof(buffer)-1 indicates truncation.
        // Implement proper framing (length-prefix or delimiter-based reads with accumulation) to avoid truncation issues.
    } else if (bytes_received == 0) {
        // Peer closed
    } else if (bytes_received == -2) {
        // Would block / timeout
    } else {
        // Fatal error
        // perror("recv");
    }
}

/*
  Usage:
  - Call handle_socket_binary(socket_fd) if treating data as binary.
  - Call handle_socket_text(socket_fd) if treating data as text.

  For full protocol correctness, implement a loop that accumulates data until a complete
  message is formed (e.g., read a fixed-size header with length, then read that many bytes,
  or read until a delimiter like '\n'), always bounding copies and validating lengths.
*/