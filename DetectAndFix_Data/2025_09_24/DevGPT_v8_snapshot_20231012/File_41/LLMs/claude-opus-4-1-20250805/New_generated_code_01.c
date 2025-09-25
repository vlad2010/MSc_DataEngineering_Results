#include <sys/socket.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

#define BUFFER_SIZE 1024

// Secure version with proper error handling and bounds checking
int handle_socket_data(int socket_fd) {
    char buffer[BUFFER_SIZE];
    ssize_t bytesReceived;
    
    // Clear the buffer to ensure no residual data
    memset(buffer, 0, sizeof(buffer));
    
    // Receive data with size limit to leave room for null terminator
    bytesReceived = recv(socket_fd, buffer, sizeof(buffer) - 1, 0);
    
    if (bytesReceived > 0) {
        // Ensure null termination (defensive programming)
        buffer[bytesReceived] = '\0';
        
        // Validate received data size
        if (bytesReceived >= BUFFER_SIZE) {
            fprintf(stderr, "Warning: Received data may be truncated\n");
            return -1;
        }
        
        // Additional input validation before processing
        // Check for malicious patterns or invalid characters
        for (ssize_t i = 0; i < bytesReceived; i++) {
            // Example: reject non-printable characters except common whitespace
            if (buffer[i] < 0x20 && buffer[i] != '\n' && 
                buffer[i] != '\r' && buffer[i] != '\t') {
                fprintf(stderr, "Invalid character detected at position %zd\n", i);
                return -1;
            }
        }
        
        // Safe data processing
        // Use bounded string operations when processing
        char processed_buffer[BUFFER_SIZE];
        strncpy(processed_buffer, buffer, sizeof(processed_buffer) - 1);
        processed_buffer[sizeof(processed_buffer) - 1] = '\0';
        
        // Process the validated data
        printf("Received %zd bytes: %s\n", bytesReceived, processed_buffer);
        
        return bytesReceived;
        
    } else if (bytesReceived == 0) {
        // Connection closed gracefully by peer
        printf("Connection closed by peer\n");
        close(socket_fd);
        return 0;
        
    } else {
        // Error occurred - check specific error type
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // Non-blocking socket, no data available
            printf("No data available (non-blocking)\n");
            return -2;
        } else if (errno == EINTR) {
            // Interrupted by signal, can retry
            printf("Receive interrupted by signal\n");
            return -3;
        } else if (errno == ECONNRESET) {
            // Connection reset by peer
            fprintf(stderr, "Connection reset by peer\n");
            close(socket_fd);
            return -4;
        } else {
            // Other error
            fprintf(stderr, "recv() failed: %s (errno: %d)\n", 
                    strerror(errno), errno);
            close(socket_fd);
            return -5;
        }
    }
}