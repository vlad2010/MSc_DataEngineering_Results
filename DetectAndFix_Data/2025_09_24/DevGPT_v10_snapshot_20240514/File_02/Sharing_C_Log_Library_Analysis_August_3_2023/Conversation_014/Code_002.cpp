FileWriter(std::string const& log_directory, std::string const& log_file_name, uint32_t log_file_roll_size_mb)
	: m_log_file_roll_size_bytes(log_file_roll_size_mb * 1024 * 1024), m_name(log_directory + log_file_name)
{
	roll_file();
}