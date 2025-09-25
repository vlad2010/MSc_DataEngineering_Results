#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <errno.h>
#include <ctype.h>
#include <sys/time.h>

#define REQ_BUFFER_SIZE 4096
#define RESP_BUFFER_SIZE 4096
#define LINE_BUFFER_SIZE 1024
#define PATH_BUFFER_SIZE 2048
#define CONNECT_TIMEOUT_SEC 5
#define IO_TIMEOUT_SEC 5

static void trim_eol(char *s) {
    if (!s) return;
    size_t len = strlen(s);
    while (len > 0) {
        unsigned char c = (unsigned char)s[len - 1];
        if (c == '\n' || c == '\r') {
            s[--len] = '\0';
        } else {
            break;
        }
    }
}

static int parse_port(const char *s, unsigned short *out_port) {
    if (!s || !*s) return -1;
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return -1;
    if (v < 1 || v > 65535) return -1;
    *out_port = (unsigned short)v;
    return 0;
}

static int has_ctl_or_space(const char *s) {
    // Disallow CTLs (<=0x1F, 0x7F) and CR/LF explicitly
    for (const unsigned char *p = (const unsigned char*)s; *p; ++p) {
        if (*p <= 0x1F || *p == 0x7F) return 1;
    }
    return 0;
}

// Percent-encode path: allow unreserved and '/' only.
// unreserved = ALPHA / DIGIT / "-" / "." / "_" / "~"
static int url_encode_path(const char *in, char *out, size_t out_size) {
    static const char hex[] = "0123456789ABCDEF";
    size_t oi = 0;

    // Ensure leading slash
    if (in[0] != '/') {
        if (out_size < 2) return -1;
        out[oi++] = '/';
    }

    for (const unsigned char *p = (const unsigned char*)in; *p; ++p) {
        unsigned char c = *p;
        int is_unreserved = (isalnum(c) || c == '-' || c == '.' || c == '_' || c == '~');
        if (is_unreserved || c == '/') {
            if (oi + 1 >= out_size) return -1;
            out[oi++] = (char)c;
        } else {
            // Encode everything else to avoid CRLF/CTL injection and ensure valid request
            if (oi + 3 >= out_size) return -1;
            out[oi++] = '%';
            out[oi++] = hex[(c >> 4) & 0xF];
            out[oi++] = hex[c & 0xF];
        }
    }
    if (oi >= out_size) return -1;
    out[oi] = '\0';
    return 0;
}

static int send_all(int fd, const char *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = send(fd, buf + total, len - total, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        total += (size_t)n;
    }
    return 0;
}

static int set_socket_timeouts(int fd, int sec) {
    struct timeval tv;
    tv.tv_sec = sec;
    tv.tv_usec = 0;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) return -1;
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) < 0) return -1;
    return 0;
}

static int connect_with_timeout(const char *host, unsigned short port) {
    struct addrinfo hints, *res = NULL, *rp = NULL;
    char portstr[8];
    snprintf(portstr, sizeof(portstr), "%u", (unsigned)port);

    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_flags    = AI_ADDRCONFIG;

    int gai = getaddrinfo(host, portstr, &hints, &res);
    if (gai != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(gai));
        return -1;
    }

    int fd = -1;
    for (rp = res; rp != NULL; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) continue;

        // Optional: set timeouts before connect; some stacks apply after connect only
        set_socket_timeouts(fd, IO_TIMEOUT_SEC);

        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            break; // success
        }
        close(fd);
        fd = -1;
    }

    freeaddrinfo(res);
    return fd; // -1 on failure
}

static int send_request(int sockfd, const char *host_for_header, const char *encoded_path) {
    char request[REQ_BUFFER_SIZE];
    int need = snprintf(request, sizeof(request),
                        "GET %s HTTP/1.1\r\n"
                        "Host: %s\r\n"
                        "User-Agent: SimpleDirBuster/1.1\r\n"
                        "Accept: */*\r\n"
                        "Connection: close\r\n\r\n",
                        encoded_path, host_for_header);
    if (need < 0 || (size_t)need >= sizeof(request)) {
        errno = EMSGSIZE;
        return -1; // truncated or error
    }
    return send_all(sockfd, request, (size_t)need);
}

static int read_status_line(int fd, char *buf, size_t bufsize) {
    // Read until '\n' or buffer full
    size_t off = 0;
    while (off + 1 < bufsize) {
        char c;
        ssize_t n = recv(fd, &c, 1, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) break; // EOF
        buf[off++] = c;
        if (c == '\n') break;
    }
    if (off == 0) return -1;
    buf[off] = '\0';
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <host_or_ip> <port> <wordlist>\n", argv[0]);
        return 1;
    }

    const char *host = argv[1];
    unsigned short port = 0;
    if (parse_port(argv[2], &port) != 0) {
        fprintf(stderr, "Invalid port: %s\n", argv[2]);
        return 1;
    }

    if (has_ctl_or_space(host)) {
        fprintf(stderr, "Invalid host (contains control characters)\n");
        return 1;
    }

    const char *wordlist_path = argv[3];
    FILE *wordlist = fopen(wordlist_path, "r");
    if (!wordlist) {
        perror("Error opening wordlist file");
        return 1;
    }

    char line[LINE_BUFFER_SIZE];
    char encoded_path[PATH_BUFFER_SIZE];

    while (fgets(line, sizeof(line), wordlist)) {
        trim_eol(line);

        // Skip empty or comment lines
        if (line[0] == '\0' || line[0] == '#') {
            continue;
        }

        // Reject lines with embedded NUL (fgets won't include), control chars will be encoded anyway;
        // Here we can also normalize whitespace; we choose to percent-encode anything non-unreserved.

        if (url_encode_path(line, encoded_path, sizeof(encoded_path)) != 0) {
            fprintf(stderr, "Path too long or encoding error for input: %s\n", line);
            continue;
        }

        int sockfd = connect_with_timeout(host, port);
        if (sockfd < 0) {
            perror("Error connecting to server");
            continue;
        }

        if (set_socket_timeouts(sockfd, IO_TIMEOUT_SEC) != 0) {
            // Not fatal; continue
        }

        if (send_request(sockfd, host, encoded_path) < 0) {
            perror("Error sending request");
            close(sockfd);
            continue;
        }

        // Read status line
        char status_line[RESP_BUFFER_SIZE];
        if (read_status_line(sockfd, status_line, sizeof(status_line)) == 0) {
            // Look for 200 OK in status line only (avoid matching in body)
            if (strstr(status_line, "HTTP/1.1 200 OK") || strstr(status_line, "HTTP/1.0 200 OK")) {
                printf("[+] Found: %s\n", line);
            }
        }
        close(sockfd);
    }

    fclose(wordlist);
    return 0;
}