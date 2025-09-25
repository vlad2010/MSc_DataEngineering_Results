#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

// Define your custom protocol header structure
struct my_protocol_header {
    uint16_t field1;
    uint32_t field2;
};

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

    // Construct your custom protocol header
    struct my_protocol_header *my_header = (struct my_protocol_header *)packet;
    my_header->field1 = htons(12345);
    my_header->field2 = htonl(54321);

    // Fill in the rest of the packet with payload data
    // You can adjust packet length and payload as needed

    // Send packet
    if (sendto(sockfd, packet, sizeof(struct my_protocol_header), 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr)) < 0) {
        perror("Sendto failed");
        exit(EXIT_FAILURE);
    }

    printf("Packet sent successfully.\n");

    // Close socket
    close(sockfd);

    return 0;
}