int main() {
    Logger logger("log.txt", LogLevel::DEBUG);
    LOG(logger, "Error message", LogLevel::ERROR);
    LOG(logger, "Warning message", LogLevel::WARNING);
    LOG(logger, "Info message", LogLevel::INFO);
    LOG(logger, "Debug message", LogLevel::DEBUG);

    return 0;
}
