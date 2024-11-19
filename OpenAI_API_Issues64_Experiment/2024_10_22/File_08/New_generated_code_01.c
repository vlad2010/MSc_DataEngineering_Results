#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>

#define BUFFER_SIZE 4096

int send_request(int sockfd, const char *host, const char *path) {
    char request[BUFFER_SIZE];
    int written = snprintf(request, sizeof(request),
                           "GET %s HTTP/1.1\r\n"
                           "Host: %s\r\n"
                           "User-Agent: SimpleDirBuster/1.0\r\n"
                           "Accept: */*\r\n"
                           "Connection: close\r\n\r\n",
                           path, host);

    if (written < 0 || written >= sizeof(request)) {
        fprintf(stderr, "Request buffer overflow.\n");
        return -1;
    }

    return send(sockfd, request, (size_t)written, 0);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <url> <port> <wordlist>\n", argv[0]);
        return 1;
    }

    const char *url = argv[1];

    char *endptr;
    long lport = strtol(argv[2], &endptr, 10);
    if (*endptr != '\0' || lport <= 0 || lport > 65535) {
        fprintf(stderr, "Invalid port number.\n");
        return 1;
    }
    int port = (int)lport;

    const char *wordlist_path = argv[3];

    FILE *wordlist = fopen(wordlist_path, "r");
    if (!wordlist) {
        perror("Error opening wordlist file");
        return 1;
    }

    char line[256];
    while (fgets(line, sizeof(line), wordlist)) {
        size_t len = strlen(line);
        if (line[len - 1] == '\n') line[len - 1] = '\0';

        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            perror("Error creating socket");
            continue;
        }

        struct addrinfo hints, *res;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        if (getaddrinfo(url, argv[2], &hints, &res) != 0) {
            perror("Error resolving hostname");
            close(sockfd);
            continue;
        }

        if (connect(sockfd, res->ai_addr, res->ai_addrlen) < 0) {
            perror("Error connecting to server");
            close(sockfd);
            freeaddrinfo(res);
            continue;
        }

        freeaddrinfo(res);

        if (send_request(sockfd, url, line) < 0) {
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
        } else if (n < 0) {
            perror("Error receiving response");
        }

        close(sockfd);
    }

    fclose(wordlist);
    return 0;
}