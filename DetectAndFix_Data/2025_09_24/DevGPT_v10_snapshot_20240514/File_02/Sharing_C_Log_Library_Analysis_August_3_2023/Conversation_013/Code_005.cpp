RingBuffer(size_t const size) : m_size(size), m_ring(static_cast<Item*>(std::malloc(size * sizeof(Item)))), m_write_index(0), m_read_index(0) {
	for (size_t i = 0; i < m_size; i++) {
		new(&m_ring[i]) Item();
	}
	static_assert(sizeof(Item) == 256, "Unexpected size != 256");
}