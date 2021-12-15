//TEMPLATE DEFINITION FOR STPFILE FOR FILENAME GENERATION
#ifdef _STP_FILE_H_

template<size_t... IP, size_t... IF, size_t... IE, typename T>
constexpr auto SuperTerrainPlus::STPFile::buildFilename(const char* path, const char* file, const char* extension,
	std::integer_sequence<T, IP...>, std::integer_sequence<T, IF...>, std::integer_sequence<T, IE...>) {
	return std::array<char, sizeof...(IP) + sizeof...(IF) + sizeof...(IE)>{ path[IP]..., file[IF]..., extension[IE]... };
}

template<size_t LP, size_t LF, size_t... LE>
constexpr auto SuperTerrainPlus::STPFile::generateFilename(const char(&path)[LP], const char(&file)[LF], const char(&...extension)[LE]) {
	using std::make_index_sequence;

	if constexpr ([](const auto& x, const auto&... xs) { return ((x == xs) && ... && true); }(LE...)) {
		//all filenames have the same size, a simple array can be used
		//each string ends with a null, so we need to eliminate the null symbol for all strings except the last one
		return std::array{
			STPFile::buildFilename(path, file, extension,
			make_index_sequence<LP - 1ull>(), make_index_sequence<LF - 1ull>(), make_index_sequence<LE>())...
		};
	}
	else {
		//to hold elements of different types, use tuple
		return std::make_tuple(
			STPFile::buildFilename(path, file, extension,
			make_index_sequence<LP - 1ull>(), make_index_sequence<LF - 1ull>(), make_index_sequence<LE>())...
		);
	}
}

#endif//_STP_FILE_H_