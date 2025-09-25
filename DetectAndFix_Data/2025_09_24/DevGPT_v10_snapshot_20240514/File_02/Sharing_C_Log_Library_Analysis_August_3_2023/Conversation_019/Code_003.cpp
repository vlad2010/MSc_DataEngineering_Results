for (size_t i = 0; i < m_size; i++) {
	new(&m_ring[i]) Item();
}