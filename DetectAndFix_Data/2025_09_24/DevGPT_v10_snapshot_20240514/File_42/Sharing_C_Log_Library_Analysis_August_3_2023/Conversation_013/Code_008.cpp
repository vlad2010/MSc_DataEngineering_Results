bool try_pop(NanoLogLine& logline)  override {
	Item& item = m_ring[m_read_index % m_size];
	SpinLock spinlock(item.flag);
	if (item.written == 1) {
		logline = std::move(item.logline);
		item.written = 0;
		++m_read_index;
		return true;
	}
	return false;
}