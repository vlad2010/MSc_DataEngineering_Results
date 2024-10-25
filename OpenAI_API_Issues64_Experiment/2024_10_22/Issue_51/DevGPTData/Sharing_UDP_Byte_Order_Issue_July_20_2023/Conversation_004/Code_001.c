#include <arpa/inet.h>

// ...

// Receive number from client
uint32_t receivedData;
recvfrom(server_socket, &receivedData, sizeof(uint32_t), 0, (struct sockaddr *)&client_addr, &client_addr_size);
uint32_t bigEndianData = ntohl(receivedData);
printf("Client: %x\n", bigEndianData);
