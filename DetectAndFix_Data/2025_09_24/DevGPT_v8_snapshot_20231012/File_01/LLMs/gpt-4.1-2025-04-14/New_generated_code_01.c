int fd = open("/dev/ttyS0", O_RDWR);
if (fd == -1) {
    perror("Failed to open device");
    return 1;
}