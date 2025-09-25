void roll_file() {
	if (m_os) {
		m_os->flush();
		m_os->close();
	}

	m_bytes_written = 0;
	m_os.reset(new std::ofstream());

	std::string log_file_name = m_name;
	log_file_name.append(".");
	log_file_name.append(std::to_string(++m_file_number));
	log_file_name.append(".txt");

	m_os->open(log_file_name, std::ofstream::out | std::ofstream::trunc);
}