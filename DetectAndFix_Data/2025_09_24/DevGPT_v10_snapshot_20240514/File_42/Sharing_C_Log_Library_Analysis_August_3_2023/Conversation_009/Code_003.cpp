	void format_timestamp(std::ostream& os, uint64_t timestamp) {
		std::time_t time_t = timestamp / 1000000;
		auto gmtime = std::gmtime(&time_t);
		char buffer[32];
		strftime(buffer, 32, "&Y-%m-%d %T.", gmtime);
		char microseconds[7];
		sprintf(microseconds, "%06llu", timestamp % 1000000);
		os << '[' << buffer << microseconds << ']';
	}