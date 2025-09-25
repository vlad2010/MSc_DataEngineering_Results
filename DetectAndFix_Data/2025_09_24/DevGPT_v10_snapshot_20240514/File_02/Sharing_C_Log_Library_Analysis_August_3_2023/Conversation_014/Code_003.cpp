void write(NanoLogLine& logline) {
	auto pos = m_os->tellp();
	logline.stringify(*m_os);
	m_bytes_written += m_os->tellp() - pos;
	if (m_bytes_written > m_log_file_roll_size_bytes) {
		roll_file();
	}
}