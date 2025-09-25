#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <errno.h>
#include <ctype.h>

#define BUFFER_SIZE 4096
#define LINE_SIZE 256

// Validate port number is numeric and in range
int validate_port(const char *port_str, int *port_out) {
    char *endptr;
    long port = strtol(port_str, &endptr, 10);
    if (*endptr != '\0' || port < 1 || port > 65535) {
        return 0;
    }
    *port_out = (int)port;
    return 1;
}

// Send HTTP request
int send_request(int sockfd, const char *host, const char *path) {
    char request[BUFFER_SIZE];
    int written = snprintf(request, sizeof(request),
             "GET %s HTTP/1.1\r\n"
             "Host: %s\r\n"
             "User-Agent: SimpleDirBuster/1.0\r\n"
             "Accept: */*\r\n"
             "Connection: close\r\n\r\n",
             path, host);
    if (written < 0 || written >= (int)sizeof(request)) {
        fprintf(stderr, "Request too long, skipping.\n");
        return -1;
    }
    return send(sockfd, request, strlen(request), 0);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <ip> <port> <wordlist>\n", argv[0]);
        return 1;
    }

    const char *ip = argv[1];
    int port;
    if (!validate_port(argv[2], &port)) {
        fprintf(stderr, "Invalid port number: %s\n", argv[2]);
        return 1;
    }
    const char *wordlist_path = argv[3];

    FILE *wordlist = fopen(wordlist_path, "r");
    if (!wordlist) {
        perror("Error opening wordlist file");
        return 1;
    }

    char line[LINE_SIZE];
    while (fgets(line, sizeof(line), wordlist)) {
        size_t len = strlen(line);

        // Warn if line is too long (no newline at end)
        if (len > 0 && line[len - 1] != '\n' && !feof(wordlist)) {
            fprintf(stderr, "Warning: line too long, skipping: %.20s...\n", line);
            // Discard rest of the line
            int ch;
            while ((ch = fgetc(wordlist)) != '\n' && ch != EOF);
            continue;
        }

        // Remove newline
        if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';

        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            perror("Error creating socket");
            continue;
        }

        struct sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);

        if (inet_pton(AF_INET, ip, &server_addr.sin_addr) <= 0) {
            perror("Error converting IP address");
            close(sockfd);
            continue;
        }

        if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            perror("Error connecting to server");
            close(sockfd);
            continue;
        }

        // Use IP as Host header, but in real-world, resolve hostname if needed
        if (send_request(sockfd, ip, line) < 0) {
            perror("Error sending request");
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

    fclose(wordlist);
    return 0;
}