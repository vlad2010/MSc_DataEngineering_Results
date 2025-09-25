#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <errno.h>

#define BUFFER_SIZE 1024

char buffer[BUFFER_SIZE];
int bytesReceived = recv(socket_fd, buffer, BUFFER_SIZE - 1, 0); // Leave space for null terminator

if (bytesReceived > 0) {
    // Null-terminate the buffer to safely use as a string
    buffer[bytesReceived] = '\0';

    // Safely process only the received data
    // Example: printf("%s\n", buffer); // Safe now

    // If you process as binary data, use bytesReceived as length
    // process_data(buffer, bytesReceived);

} else if (bytesReceived == 0) {
    // Connection closed
} else {
    // Error occurred
    perror("recv failed");
    // Handle error, e.g., check errno
}