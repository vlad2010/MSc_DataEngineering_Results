// 블로킹 모드로 소켓 설정
int flags = fcntl(socket_fd, F_GETFL, 0);
flags &= ~O_NONBLOCK;
fcntl(socket_fd, F_SETFL, flags);

// 이후에 select() 또는 poll() 호출
