int main() {
    Logger logger;
    logger.log("Error message", __FILE__, __LINE__, LogLevel::ERROR);
    logger.log("Warning message", __FILE__, __LINE__, LogLevel::WARNING);
    logger.log("Info message", __FILE__, __LINE__, LogLevel::INFO);
    logger.log("Debug message", __FILE__, __LINE__, LogLevel::DEBUG);

    return 0;
}
