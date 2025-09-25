	while (m_state.load() == State::READY) {
		if (m_buffer_base->try_pop(logline)) {
			m_file_writer.write(logline);
		}
		else {
			std::this_thread::sleep_for(std::chrono::microseconds(50));
		}
	}