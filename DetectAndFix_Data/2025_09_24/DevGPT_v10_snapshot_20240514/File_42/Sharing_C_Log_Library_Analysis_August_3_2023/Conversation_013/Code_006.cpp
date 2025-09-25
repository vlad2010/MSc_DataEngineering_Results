~RingBuffer() {
	for (size_t i = 0; i < m_size; i++) {
		m_ring[i].~Item();
	}
	std::free(m_ring);
}