void push(NanoLogLine&& logline)  override {
	unsigned int write_index = m_write_index.fetch_add(1, std::memory_order_relaxed) % m_size;
	Item& item = m_ring[write_index];
	SpinLock spinlock(item.flag);
	item.logline = std::move(logline);
	item.written = 1;
}