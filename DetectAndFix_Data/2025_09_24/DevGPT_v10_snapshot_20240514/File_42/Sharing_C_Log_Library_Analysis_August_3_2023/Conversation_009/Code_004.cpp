	std::thread::id this_thread_id() {
		static thread_local const std::thread::id id = std::this_thread::get_id();
		return id;
	}