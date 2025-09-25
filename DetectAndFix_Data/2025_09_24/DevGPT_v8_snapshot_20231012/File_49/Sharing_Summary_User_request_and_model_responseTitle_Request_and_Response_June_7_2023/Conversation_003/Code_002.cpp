int main() {
    Logger logger("log.txt");
    logger.log("Program started.", __FILE__, __LINE__);

    // ... Your program code ...

    logger.log("Program finished.", __FILE__, __LINE__);
    return 0;
}
