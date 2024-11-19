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

#define BUFFER_SIZE 4096
#define NUM_THREADS 10

char url[BUFFER_SIZE];
int port;
pthread_mutex_t wordlist_mutex;
FILE *wordlist;

// Utility function to validate if a string is a valid IP address
int is_valid_ip(const char *ip) {
    struct sockaddr_in sa;
    return inet_pton(AF_INET, ip, &(sa.sin_addr)) != 0;
}

int send_request(int sockfd, const char *host, const char *path) {
    char request[BUFFER_SIZE];
    int request_length = snprintf(request, sizeof(request),
             "GET %s HTTP/1.1\r\n"
             "Host: %s\r\n"
             "User-Agent: SimpleDirBuster/1.0\r\n"
             "Accept: */*\r\n"
             "Connection: close\r\n\r\n",
             path, host);

    if (request_length < 0 || request_length >= sizeof(request)) {
        fprintf(stderr, "Request too long!\n");
        return -1;
    }

    return send(sockfd, request, strlen(request), 0);
}

void *dirbuster_thread(void *arg) {
    char line[256];

    while (1) {
        pthread_mutex_lock(&wordlist_mutex);
        if (fgets(line, sizeof(line), wordlist) == NULL) {
            pthread_mutex_unlock(&wordlist_mutex);
            break;
        }

        size_t len = strlen(line);
        if (line[len - 1] == '\n') line[len - 1] = '\0';
        pthread_mutex_unlock(&wordlist_mutex);

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

    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <url> <port> <wordlist>\n", argv[0]);
        return 1;
    }

    strncpy(url, argv[1], BUFFER_SIZE - 1);
    url[BUFFER_SIZE - 1] = '\0';

    if (!is_valid_ip(url)) {
        fprintf(stderr, "Invalid IP address: %s\n", url);
        return 1;
    }

    port = atoi(argv[2]);
    if (port <= 0 || port > 65535) {
        fprintf(stderr, "Invalid port number: %s\n", argv[2]);
        return 1;
    }

    const char *wordlist_path = argv[3];
    wordlist = fopen(wordlist_path, "r");
    if (!wordlist) {
        perror("Error opening wordlist file");
        return 1;
    }

    if (pthread_mutex_init(&wordlist_mutex, NULL) != 0) {
        perror("Mutex initialization failed");
        fclose(wordlist);
        return 1;
    }

    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_create(&threads[i], NULL, dirbuster_thread, NULL) != 0) {
            fprintf(stderr, "Error creating thread %d\n", i);
            fclose(wordlist);
            pthread_mutex_destroy(&wordlist_mutex);
            return 1;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    fclose(wordlist);
    pthread_mutex_destroy(&wordlist_mutex);
    return 0;
}