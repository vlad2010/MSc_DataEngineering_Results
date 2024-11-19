fd_set readfds;
FD_ZERO(&readfds);
FD_SET(socket_fd, &readfds);

struct timeval timeout;
timeout.tv_sec = 5;  // 타임아웃 시간 설정
timeout.tv_usec = 0;

int ready = select(socket_fd + 1, &readfds, NULL, NULL, &timeout);
if (ready > 0) {
    // 소켓에서 데이터가 도착함
} else if (ready == 0) {
    // 타임아웃 발생
} else {
    // 에러 발생
}
