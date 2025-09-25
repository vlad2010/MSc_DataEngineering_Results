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
#include <errno.h>
#include <ctype.h>

#define BUFFER_SIZE 4096
#define NUM_THREADS 10
#define MAX_LINE_LENGTH 255

// Helper function to check for CRLF injection
int contains_crlf(const char *s) {
    while (*s) {
        if (*s == '\r' || *s == '\n') return 1;
        s++;
    }
    return 0;
}

// Helper function to validate port
int parse_port(const char *s) {
    char *endptr;
    long port = strtol(s, &endptr, 10);
    if (*endptr != '\0' || port < 1 || port > 65535) {
        return -1;
    }
    return (int)port;
}

// Read wordlist into memory for thread safety
typedef struct {
    char **lines;
    size_t count;
    size_t next;
    pthread_mutex_t mutex;
} wordlist_t;

wordlist_t wordlist;

const char *url;
int port;

int send_request(int sockfd, const char *host, const char *path) {
    char request[BUFFER_SIZE];

    // Prevent HTTP header injection
    if (contains_crlf(host) || contains_crlf(path)) {
        fprintf(stderr, "CRLF injection detected in host or path\n");
        return -1;
    }

    int written = snprintf(request, sizeof(request),
             "GET %s HTTP/1.1\r\n"
             "Host: %s\r\n"
             "User-Agent: SimpleDirBuster/1.0\r\n"
             "Accept: */*\r\n"
             "Connection: close\r\n\r\n",
             path, host);

    if (written < 0 || (size_t)written >= sizeof(request)) {
        fprintf(stderr, "Request too large\n");
        return -1;
    }

    return send(sockfd, request, strlen(request), 0);
}

void *dirbuster_thread(void *arg) {
    while (1) {
        char *line = NULL;

        // Thread-safe access to wordlist
        pthread_mutex_lock(&wordlist.mutex);
        if (wordlist.next < wordlist.count) {
            line = wordlist.lines[wordlist.next++];
        }
        pthread_mutex_unlock(&wordlist.mutex);

        if (!line) break;

        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            perror("Error creating socket");
            continue;
        }

        struct sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);

        if (inet_pton(AF_INET, url, &server_addr.sin_addr) <= 0) {
            perror("Error converting IP address");
            close(sockfd);
            continue;
        }

        if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            perror("Error connecting to server");
            close(sockfd);
            continue;
        }

        if (send_request(sockfd, url, line) < 0) {
            // Error already printed
            close(sockfd);
            continue;
        }

        char response[BUFFER_SIZE];
        ssize_t n = recv(sockfd, response, sizeof(response) - 1, 0);
        if (n > 0) {
            response[n] = '\0';
            if (strstr(response, "HTTP/1.1 200 OK") || strstr(response, "HTTP/1.0 200 OK")) {
                printf("[+] Found: %s\n", line);
            }
        }

        close(sockfd);
    }

    return NULL;
}

// Helper to trim newline and carriage return
void trim_newline(char *s) {
    size_t len = strlen(s);
    while (len > 0 && (s[len-1] == '\n' || s[len-1] == '\r')) {
        s[--len] = '\0';
    }
}

// Load wordlist into memory
int load_wordlist(const char *path, wordlist_t *wl) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    size_t cap = 1024;
    wl->lines = malloc(cap * sizeof(char *));
    if (!wl->lines) {
        fclose(f);
        return -1;
    }
    wl->count = 0;
    wl->next = 0;
    pthread_mutex_init(&wl->mutex, NULL);

    char buf[MAX_LINE_LENGTH + 2]; // +2 for newline and null
    while (fgets(buf, sizeof(buf), f)) {
        trim_newline(buf);
        size_t len = strlen(buf);
        if (len == 0) continue;
        if (len > MAX_LINE_LENGTH) {
            fprintf(stderr, "Warning: line too long, truncated: %s\n", buf);
            buf[MAX_LINE_LENGTH] = '\0';
        }
        if (wl->count == cap) {
            cap *= 2;
            char **tmp = realloc(wl->lines, cap * sizeof(char *));
            if (!tmp) {
                fclose(f);
                return -1;
            }
            wl->lines = tmp;
        }
        wl->lines[wl->count] = strdup(buf);
        if (!wl->lines[wl->count]) {
            fclose(f);
            return -1;
        }
        wl->count++;
    }
    fclose(f);
    return 0;
}

void free_wordlist(wordlist_t *wl) {
    for (size_t i = 0; i < wl->count; ++i) {
        free(wl->lines[i]);
    }
    free(wl->lines);
    pthread_mutex_destroy(&wl->mutex);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <ip> <port> <wordlist>\n", argv[0]);
        return 1;
    }

    url = argv[1];
    port = parse_port(argv[2]);
    if (port == -1) {
        fprintf(stderr, "Invalid port: %s\n", argv[2]);
        return 1;
    }

    // Validate IP address
    struct in_addr addr;
    if (inet_pton(AF_INET, url, &addr) <= 0) {
        fprintf(stderr, "Invalid IPv4 address: %s\n", url);
        return 1;
    }

    if (load_wordlist(argv[3], &wordlist) != 0) {
        perror("Error loading wordlist file");
        return 1;
    }

    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_create(&threads[i], NULL, dirbuster_thread, NULL) != 0) {
            perror("Error creating thread");
            // Clean up and exit
            for (int j = 0; j < i; ++j) pthread_join(threads[j], NULL);
            free_wordlist(&wordlist);
            return 1;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    free_wordlist(&wordlist);
    return 0;
}