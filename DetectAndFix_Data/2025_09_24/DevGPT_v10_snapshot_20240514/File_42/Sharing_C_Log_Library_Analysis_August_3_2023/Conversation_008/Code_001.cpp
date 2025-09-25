#include "NanoLog.hpp"
#include <cerrno>
#include <cstring>
#include <chrono>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

// 包含头文件和库。这些库提供了许多在代码中需要的功能，如时间处理（<chrono>、<ctime>）、文件操作（<fstream>）、多线程支持（<thread>、<mutex>）等。

namespace nanolog
{
    // 将所有的代码放在 nanolog 命名空间下，以避免命名冲突。
    
    enum class LogLevel : uint8_t { INFO, WARN, CRIT };
    // 定义了一个名为 LogLevel 的枚举类，用于表示日志级别。枚举值包括 INFO、WARN 和 CRIT。

    char const * to_string(LogLevel loglevel)
    {
        // 这个函数用于将 LogLevel 枚举值转换为对应的字符串。
        
        switch (loglevel)
        {
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARN: return "WARN";
            case LogLevel::CRIT: return "CRIT";
        }
        return "UNKNOWN";
    }