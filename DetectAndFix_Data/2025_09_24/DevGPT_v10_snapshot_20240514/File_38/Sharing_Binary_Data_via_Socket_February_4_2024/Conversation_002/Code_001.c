#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#define DEST_IP "127.0.0.1"
#define DEST_PORT 8080

int main() {
    int sockfd;
    struct sockaddr_in dest_addr;
    char packet[4096]; // Maximum size of the packet

    // Create socket
    sockfd = socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
    if (sockfd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Initialize destination address structure
    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_addr.s_addr = inet_addr(DEST_IP);

    // Construct IP header
    struct iphdr *ip_header = (struct iphdr *)packet;
    ip_header->ihl = 5; // Internet Header Length
    ip_header->version = 4; // IPv4
    ip_header->tos = 0; // Type of Service
    ip_header->tot_len = htons(sizeof(struct iphdr) + sizeof(struct tcphdr)); // Total length
    ip_header->id = htons(54321); // Identification
    ip_header->frag_off = 0; // Fragment Offset
    ip_header->ttl = 255; // Time to Live
    ip_header->protocol = IPPROTO_TCP; // Protocol (TCP)
    ip_header->check = 0; // Checksum (to be calculated later)
    ip_header->saddr = inet_addr("192.168.1.100"); // Source IP address
    ip_header->daddr = dest_addr.sin_addr.s_addr; // Destination IP address

    // Construct TCP header
    struct tcphdr *tcp_header = (struct tcphdr *)(packet + sizeof(struct iphdr));
    tcp_header->source = htons(12345); // Source port
    tcp_header->dest = htons(DEST_PORT); // Destination port
    tcp_header->seq = 0; // Sequence number
    tcp_header->ack_seq = 0; // Acknowledgment number
    tcp_header->doff = 5; // Data offset
    tcp_header->fin = 0;
    tcp_header->syn = 1; // SYN flag
    tcp_header->rst = 0;
    tcp_header->psh = 0;
    tcp_header->ack = 0;
    tcp_header->urg = 0;
    tcp_header->window = htons(5840); // Window size
    tcp_header->check = 0; // Checksum (to be calculated later)
    tcp_header->urg_ptr = 0; // Urgent pointer

    // Calculate TCP checksum
    // You need to calculate IP header checksum as well but omitted here for simplicity

    // Send packet
    if (sendto(sockfd, packet, sizeof(struct iphdr) + sizeof(struct tcphdr), 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr)) < 0) {
        perror("Sendto failed");
        exit(EXIT_FAILURE);
    }

    printf("Packet sent successfully.\n");

    // Close socket
    close(sockfd);

    return 0;
}