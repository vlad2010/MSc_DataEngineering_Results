struct timeval timeout;
timeout.tv_sec = 0;
timeout.tv_usec = 0;

int ready = select(socket_fd + 1, &readfds, NULL, NULL, &timeout);
if (ready > 0) {
    // 데이터가 있는 경우 처리
}
