Item& item = m_ring[m_read_index % m_size];
SpinLock spinlock(item.flag);