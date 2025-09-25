~SpinLock() {
	m_flag.clear(std::memory_order_release); // 使用内存屏障，使得别的线程知道该锁的消息
}