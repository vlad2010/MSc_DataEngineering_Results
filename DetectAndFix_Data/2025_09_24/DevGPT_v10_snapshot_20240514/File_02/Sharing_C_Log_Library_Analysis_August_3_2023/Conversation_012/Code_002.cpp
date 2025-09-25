	while (m_state.load(std::memory_order_acquire) == State::INIT) {
		std::this_thread::sleep_for(std::chrono::microseconds(50));
	}