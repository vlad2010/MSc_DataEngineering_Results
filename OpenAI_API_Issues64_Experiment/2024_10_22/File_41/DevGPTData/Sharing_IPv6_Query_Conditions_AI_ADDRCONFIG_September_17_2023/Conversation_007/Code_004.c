struct pollfd fds;
fds.fd = socket_fd;
fds.events = POLLIN;  // 읽을 데이터 확인
fds.revents = 0;

int ready = poll(&fds, 1, 5000);  // 5초 타임아웃
if (ready > 0) {
    // 소켓에서 데이터를 읽을 수 있음
}
