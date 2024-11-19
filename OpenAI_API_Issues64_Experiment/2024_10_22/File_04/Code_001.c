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

#define BUFFER_SIZE 4096
#define NUM_THREADS 10

const char *url;
int port;
pthread_mutex_t wordlist_mutex;
FILE *wordlist;

int send_request(int sockfd, const char *host, const char *path) {
    char request[BUFFER_SIZE];
    snprintf(request, sizeof(request),
             "GET %s HTTP/1.1\r\n"
             "Host: %s\r\n"
             "User-Agent: SimpleDirBuster/1.0\r\n"
             "Accept: */*\r\n"
             "Connection: close\r\n\r\n",
             path, host);

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
        pthread_mutex_unlock(&wordlist_mutex);

        size_t len = strlen(line);
        if (line[len - 1] == '\n') line[len - 1] = '\0';

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

    url = argv[1];
    port = atoi(argv[2]);
    const char *wordlist_path = argv[3];

    wordlist = fopen(wordlist_path, "r");
    if (!wordlist) {
        perror("Error opening wordlist file");
        return 1;
    }

    pthread_mutex_init(&wordlist_mutex, NULL);

    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_create(&threads[i], NULL, dirbuster_thread, NULL) != 0) {
           
