int flags = fcntl(socket_fd, F_GETFL, 0);
flags |= O_NONBLOCK;
fcntl(socket_fd, F_SETFL, flags);
