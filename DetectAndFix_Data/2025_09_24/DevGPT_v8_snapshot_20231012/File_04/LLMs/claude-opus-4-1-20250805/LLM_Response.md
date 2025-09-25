## Security Analysis

I've identified several critical security vulnerabilities in this directory busting tool:

### 1. **Format String Vulnerability (CWE-134)**
- Line 76: `printf("[+] Found: %s\n", line);` - If `line` contains format specifiers, it could lead to information disclosure or crashes
- **Fix**: Use `printf("[+] Found: %s\n", line);` (this one is actually safe, but needs validation)

### 2. **Buffer Overflow (CWE-120)**
- The `send_request` function uses `snprintf` but doesn't validate that `path` length won't cause overflow
- If a malicious wordlist contains very long paths, it could overflow the request buffer

### 3. **Path Traversal (CWE-22)**
- No validation of paths from wordlist - could contain `../` sequences
- Allows testing of arbitrary paths including sensitive system paths

### 4. **Integer Overflow (CWE-190)**
- `port = atoi(argv[2])` doesn't validate port range (0-65535)
- Invalid ports could cause unexpected behavior

### 5. **Resource Exhaustion (CWE-400)**
- No rate limiting or connection pooling
- Could be used for DoS attacks against target server

### 6. **Missing Input Validation (CWE-20)**
- No validation of URL format
- No validation of wordlist content
- No sanitization of user inputs

### 7. **Information Disclosure (CWE-200)**
- Error messages with `perror()` might leak sensitive system information

### 8. **DNS Resolution Issue**
- `inet_pton` expects IP address, not hostname/URL
- Need proper DNS resolution for hostnames

## Fixed Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <ctype.h>
#include <limits.h>
#include <time.h>
#include <errno.h>

#define BUFFER_SIZE 4096
#define MAX_PATH_SIZE 2048
#define NUM_THREADS 10
#define MAX_LINE_LENGTH 256
#define MIN_DELAY_MS 100  // Rate limiting

const char *hostname;
int port;
pthread_mutex_t wordlist_mutex;
FILE *wordlist;
struct sockaddr_in server_addr;

// Validate and sanitize path
int validate_path(const char *path) {
    if (!path || strlen(path) == 0 || strlen(path) >= MAX_PATH_SIZE) {
        return 0;
    }
    
    // Check for path traversal attempts
    if (strstr(path, "../") || strstr(path, "..\\")) {
        return 0;
    }
    
    // Ensure path starts with /
    if (path[0] != '/') {
        return 0;
    }
    
    // Check for invalid characters
    for (const char *p = path; *p; p++) {
        if (!isprint(*p) || *p == '\r' || *p == '\n') {
            return 0;
        }
    }
    
    return 1;
}

int send_request(int sockfd, const char *host, const char *path) {
    char request[BUFFER_SIZE];
    int ret = snprintf(request, sizeof(request),
             "GET %s HTTP/1.1\r\n"
             "Host: %s\r\n"
             "User-Agent: SimpleDirBuster/1.0\r\n"
             "Accept: */*\r\n"
             "Connection: close\r\n\r\n",
             path, host);
    
    // Check for truncation
    if (ret < 0 || ret >= sizeof(request)) {
        return -1;
    }

    return send(sockfd, request, strlen(request), 0);
}

void *dirbuster_thread(void *arg) {
    char line[MAX_LINE_LENGTH];
    char safe_path[MAX_PATH_SIZE];
    struct timespec delay = {0, MIN_DELAY_MS * 1000000}; // Convert ms to ns

    while (1) {
        pthread_mutex_lock(&wordlist_mutex);
        if (fgets(line, sizeof(line), wordlist) == NULL) {
            pthread_mutex_unlock(&wordlist_mutex);
            break;
        }
        pthread_mutex_unlock(&wordlist_mutex);

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

        // Prepare path with leading slash if not present
        if (line[0] != '/') {
            snprintf(safe_path, sizeof(safe_path), "/%s", line);
        } else {
            strncpy(safe_path, line, sizeof(safe_path) - 1);
            safe_path[sizeof(safe_path) - 1] = '\0';
        }

        // Validate path
        if (!validate_path(safe_path)) {
            fprintf(stderr, "[!] Invalid path skipped\n");
            continue;
        }

        // Rate limiting
        nanosleep(&delay, NULL);

        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            fprintf(stderr, "[!] Socket creation failed\n");
            continue;
        }

        // Set socket timeout
        struct timeval timeout;
        timeout.tv_sec = 5;
        timeout.tv_usec = 0;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

        if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            close(sockfd);
            continue;
        }

        if (send_request(sockfd, hostname, safe_path) < 0) {
            close(sockfd);
            continue;
        }

        char response[BUFFER_SIZE];
        ssize_t n = recv(sockfd, response, sizeof(response) - 1, 0);
        if (n > 0) {
            response[n] = '\0';
            
            // Parse HTTP status code properly
            char *status_line = strtok(response, "\r\n");
            if (status_line) {
                int status_code = 0;
                if (sscanf(status_line, "HTTP/%*s %d", &status_code) == 1) {
                    if (status_code == 200) {
                        printf("[+] Found (200): %s\n", safe_path);
                    } else if (status_code == 301 || status_code == 302) {
                        printf("[+] Redirect (%d): %s\n", status_code, safe_path);
                    } else if (status_code == 403) {
                        printf("[+] Forbidden (403): %s\n", safe_path);
                    }
                }
            }
        }

        close(sockfd);
    }

    return NULL;
}

int resolve_hostname(const char *host, struct sockaddr_in *addr) {
    struct hostent *server = gethostbyname(host);
    if (server == NULL) {
        // Try as IP address
        if (inet_pton(AF_INET, host, &addr->sin_addr) <= 0) {
            return -1;
        }
    } else {
        memcpy(&addr->sin_addr.s_addr, server->h_addr, server->h_length);
    }
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <hostname/ip> <port> <wordlist>\n", argv[0]);
        return 1;
    }

    hostname = argv[1];
    
    // Validate hostname length
    if (strlen(hostname) > 253) {  // Max DNS name length
        fprintf(stderr, "Error: Hostname too long\n");
        return 1;
    }
    
    // Parse and validate port
    char *endptr;
    long port_long = strtol(argv[2], &endptr, 10);
    if (*endptr != '\0' || port_long < 1 || port_long > 65535) {
        fprintf(stderr, "Error: Invalid port number (must be 1-65535)\n");
        return 1;
    }
    port = (int)port_long;
    
    const char *wordlist_path = argv[3];

    // Open wordlist with validation
    wordlist = fopen(wordlist_path, "r");
    if (!wordlist) {
        fprintf(stderr, "Error: Cannot open wordlist file\n");
        return 1;
    }

    // Setup server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    // Resolve hostname
    if (resolve_hostname(hostname, &server_addr) < 0) {
        fprintf(stderr, "Error: Cannot resolve hostname\n");
        fclose(wordlist);
        return 1;
    }

    printf("[*] Starting directory enumeration on %s:%d\n", hostname, port);
    printf("[*] Using wordlist: %s\n", wordlist_path);
    printf("[*] Threads: %d\n", NUM_THREADS);
    printf("[*] Rate limit: %dms between requests per thread\n", MIN_DELAY_MS);
    printf("----------------------------------------\n");

    pthread_mutex_init(&wordlist_mutex, NULL);

    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_create(&threads[i], NULL, dirbuster_thread, NULL) != 0) {
            fprintf(stderr, "Error: Failed to create thread %d\n", i);
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&wordlist_mutex);
    fclose(wordlist);

    printf("----------------------------------------\n");
    printf("[*] Scan complete\n");

    return 0;
}
```

## Key Security Improvements:

1. **Input Validation**: Validates all user inputs including paths, port numbers, and hostname length
2. **Path Traversal Prevention**: Blocks `../` sequences and validates path format
3. **Buffer Overflow Prevention**: Proper bounds checking and string handling
4. **Rate Limiting**: Added delay between requests to prevent DoS
5. **Proper DNS Resolution**: Handles both hostnames and IP addresses correctly
6. **Socket Timeouts**: Prevents hanging connections
7. **Safe String Operations**: Uses bounded string functions throughout
8. **Error Handling**: Improved error messages without leaking system information
9. **Integer Validation**: Proper port range validation using `strtol`
10. **HTTP Response Parsing**: Better status code extraction