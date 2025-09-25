	while (m_buffer_base->try_pop(logline)) {
		m_file_writer.write(logline);
	}