std::atomic_flag flag;
char written;
char padding[256 - sizeof(std::atomic_flag) - sizeof(char) - sizeof(NanoLogLine)];
NanoLogLine logline;