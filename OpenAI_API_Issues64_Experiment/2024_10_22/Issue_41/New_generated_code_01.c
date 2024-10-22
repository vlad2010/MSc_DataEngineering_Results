#include <sys/socket.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#define BUFFER_SIZE 1024

char buffer[BUFFER_SIZE];
int bytesReceived = recv(socket_fd, buffer, BUFFER_SIZE - 1, 0); // Reserve one byte for null-termination

if (bytesReceived > 0) {
    // Null-terminate to safely use buffer as a string (if needed for string operations)
    buffer[bytesReceived] = '\0';

    // Process the data in buffer here
} else if (bytesReceived == 0) {
    // Connection has been gracefully closed
    printf("Connection closed\n");
} else {
    // An error occurred, or the connection was disrupted
    perror("recv failed");
    // You may handle specific errors with code like:
    // if (errno == EINTR) { ... } else if (errno == EAGAIN) { ... } etc.
}