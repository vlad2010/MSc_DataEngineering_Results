fd_set readfds;
struct timeval timeout;
FD_ZERO(&readfds);
FD_SET(socket_fd, &readfds);
timeout.tv_sec = 5;  // 5초 타임아웃
timeout.tv_usec = 0;

int ready = select(socket_fd + 1, &readfds, NULL, NULL, &timeout);
if (ready > 0) {
    // 소켓에서 데이터를 읽을 수 있음
}
