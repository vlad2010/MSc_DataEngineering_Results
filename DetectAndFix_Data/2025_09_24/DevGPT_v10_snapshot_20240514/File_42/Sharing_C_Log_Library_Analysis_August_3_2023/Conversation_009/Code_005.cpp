	template <typename T, typename Tuple>
	struct TupleIndex;

	template <typename T, typename ... Types>
	struct TupleIndex <T, std::tuple<T, Types...>> {
		static constexpr const std::size_t value = 0;
	};

	template <typename T,typename U ,typename ... Types>
	struct TupleIndex <T, std::tuple<U, Types...>> {
		static constexpr const std::size_t value = 1 + TupleIndex < T, std::tuple < Types... > >::value;
	};