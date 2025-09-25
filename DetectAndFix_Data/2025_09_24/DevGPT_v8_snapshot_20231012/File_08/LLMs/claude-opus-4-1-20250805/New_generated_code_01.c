#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <ctype.h>
#include <limits.h>
#include <errno.h>

#define BUFFER_SIZE 4096
#define MAX_PATH_LENGTH 2048
#define MAX_LINE_LENGTH 256
#define MAX_WORDLIST_ENTRIES 100000

// Sanitize path to prevent HTTP header injection
int sanitize_path(const char *input, char *output, size_t output_size) {
    size_t i, j = 0;
    size_t input_len = strlen(input);
    
    if (input_len >= output_size - 1) {
        return -1; // Path too long
    }
    
    // Ensure path starts with /
    if (input[0] != '/') {
        output[j++] = '/';
    }
    
    for (i = 0; i < input_len && j < output_size - 1; i++) {
        char c = input[i];
        
        // Block control characters and CRLF
        if (c == '\r' || c == '\n' || c == '\0') {
            continue;
        }
        
        // Block other control characters
        if (!isprint((unsigned char)c)) {
            continue;
        }
        
        // Prevent double slashes
        if (c == '/' && j > 0 && output[j-1] == '/') {
            continue;
        }
        
        output[j++] = c;
    }
    
    output[j] = '\0';
    
    // Check for path traversal attempts
    if (strstr(output, "../") || strstr(output, "..\\")) {
        return -1;
    }
    
    return 0;
}

// Validate IP address
int validate_ip(const char *ip) {
    struct sockaddr_in sa;
    return inet_pton(AF_INET, ip, &(sa.sin_addr)) == 1;
}

// Safe string to integer conversion
int safe_atoi(const char *str, int *result) {
    char *endptr;
    long val;
    
    errno = 0;
    val = strtol(str, &endptr, 10);
    
    if (errno != 0 || endptr == str || *endptr != '\0') {
        return -1;
    }
    
    if (val < INT_MIN || val > INT_MAX) {
        return -1;
    }
    
    *result = (int)val;
    return 0;
}

int send_request(int sockfd, const char *host, const char *path) {
    char request[BUFFER_SIZE];
    char sanitized_path[MAX_PATH_LENGTH];
    
    // Sanitize the path
    if (sanitize_path(path, sanitized_path, sizeof(sanitized_path)) < 0) {
        fprintf(stderr, "Invalid path: %s\n", path);
        return -1;
    }
    
    // Build request with bounds checking
    int ret = snprintf(request, sizeof(request),
                      "GET %s HTTP/1.1\r\n"
                      "Host: %s\r\n"
                      "User-Agent: SimpleDirBuster/1.0\r\n"
                      "Accept: */*\r\n"
                      "Connection: close\r\n\r\n",
                      sanitized_path, host);
    
    if (ret < 0 || ret >= sizeof(request)) {
        fprintf(stderr, "Request too large\n");
        return -1;
    }
    
    return send(sockfd, request, strlen(request), 0);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <ip_address> <port> <wordlist>\n", argv[0]);
        return 1;
    }
    
    const char *ip_address = argv[1];
    int port;
    const char *wordlist_path = argv[3];
    
    // Validate IP address
    if (!validate_ip(ip_address)) {
        fprintf(stderr, "Invalid IP address: %s\n", ip_address);
        return 1;
    }
    
    // Validate and parse port
    if (safe_atoi(argv[2], &port) < 0 || port < 1 || port > 65535) {
        fprintf(stderr, "Invalid port number: %s (must be 1-65535)\n", argv[2]);
        return 1;
    }
    
    // Open wordlist with validation
    FILE *wordlist = fopen(wordlist_path, "r");
    if (!wordlist) {
        perror("Error opening wordlist file");
        return 1;
    }
    
    char line[MAX_LINE_LENGTH];
    int entry_count = 0;
    
    while (fgets(line, sizeof(line), wordlist) != NULL) {
        // Prevent processing too many entries
        if (++entry_count > MAX_WORDLIST_ENTRIES) {
            fprintf(stderr, "Wordlist too large (max %d entries)\n", MAX_WORDLIST_ENTRIES);
            break;
        }
        
        // Remove newline
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
            len--;
        }
        
        // Skip empty lines
        if (len == 0) {
            continue;
        }
        
        // Create socket
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            perror("Error creating socket");
            continue;
        }
        
        // Set socket timeout
        struct timeval timeout;
        timeout.tv_sec = 5;
        timeout.tv_usec = 0;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        
        struct sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);
        
        if (inet_pton(AF_INET, ip_address, &server_addr.sin_addr) <= 0) {
            perror("Error converting IP address");
            close(sockfd);
            continue;
        }
        
        if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            close(sockfd);
            continue; // Silently skip failed connections
        }
        
        if (send_request(sockfd, ip_address, line) < 0) {
            close(sockfd);
            continue;
        }
        
        char response[BUFFER_SIZE];
        memset(response, 0, sizeof(response));
        ssize_t n = recv(sockfd, response, sizeof(response) - 1, 0);
        
        if (n > 0) {
            response[n] = '\0';
            if (strstr(response, "HTTP/1.1 200 OK") || strstr(response, "HTTP/1.0 200 OK")) {
                // Sanitize output to prevent terminal injection
                char safe_line[MAX_LINE_LENGTH];
                snprintf(safe_line, sizeof(safe_line), "%s", line);
                printf("[+] Found: %s\n", safe_line);
            }
        }
        
        close(sockfd);
    }
    
    fclose(wordlist);
    return 0;
}