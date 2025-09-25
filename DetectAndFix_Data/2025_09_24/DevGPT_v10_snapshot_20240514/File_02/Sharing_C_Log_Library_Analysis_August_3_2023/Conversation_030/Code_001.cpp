struct SpinLock {
	SpinLock(std::atomic_flag& flag) : m_flag(flag) {
		while (m_flag.test_and_set(std::memory_order_acquire)); // 通过 while 循环实现忙等待，直到获取锁
	}
	~SpinLock() {
		m_flag.clear(std::memory_order_release); // 使用内存屏障，使得别的线程知道该锁的消息
	}
private:
	std::atomic_flag& m_flag; // 一个布尔标识
};