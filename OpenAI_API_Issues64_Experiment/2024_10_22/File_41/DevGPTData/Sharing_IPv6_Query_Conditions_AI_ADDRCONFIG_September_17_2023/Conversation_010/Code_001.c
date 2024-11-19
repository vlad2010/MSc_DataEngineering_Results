char buffer[1024];
int bytesReceived = recv(socket_fd, buffer, sizeof(buffer), 0);

if (bytesReceived > 0) {
    // 데이터가 도착함, buffer에 있는 데이터를 처리
} else if (bytesReceived == 0) {
    // 연결이 종료됨
} else {
    // 오류가 발생했거나 연결이 중단됨
}
