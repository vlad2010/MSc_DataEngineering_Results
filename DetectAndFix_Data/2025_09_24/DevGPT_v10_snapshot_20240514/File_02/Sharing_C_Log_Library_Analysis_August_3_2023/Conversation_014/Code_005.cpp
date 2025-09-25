uint32_t m_file_number = 0;
std::streamoff m_bytes_written = 0;
uint32_t const m_log_file_roll_size_bytes;
std::string const m_name;
std::unique_ptr<std::ofstream> m_os;