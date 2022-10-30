//TEMPLATE DEFINITION FOR STPFILE FOR FILENAME GENERATION
#ifdef _STP_FILE_H_

template<size_t... IP, size_t... IF, size_t... IE, typename T>
inline constexpr auto SuperTerrainPlus::STPFile::STPFileImpl::buildFilename(const char* path, const char* file, const char* extension,
	std::integer_sequence<T, IP...>, std::integer_sequence<T, IF...>, std::integer_sequence<T, IE...>) {
	return std::array<char, sizeof...(IP) + sizeof...(IF) + sizeof...(IE)>{ path[IP]..., file[IF]..., extension[IE]... };
}

template<size_t LP, size_t LF, size_t... LE>
inline constexpr auto SuperTerrainPlus::STPFile::generateFilename(const char(&path)[LP], const char(&file)[LF], const char(&...extension)[LE]) {
	using std::make_index_sequence;
	constexpr auto allSameSize = [](const auto& x, const auto&... xs) constexpr -> bool {
		return ((x == xs) && ... && true);
	};

	if constexpr (allSameSize(LE...)) {
		//all filenames have the same size, a simple array can be used
		//each string ends with a null, so we need to eliminate the null symbol for all strings except the last one
		return std::array{
			STPFileImpl::buildFilename(path, file, extension,
			make_index_sequence<LP - 1ull>(), make_index_sequence<LF - 1ull>(), make_index_sequence<LE>())...
		};
	}
	else {
		//to hold elements of different types, use tuple
		return std::make_tuple(
			STPFileImpl::buildFilename(path, file, extension,
			make_index_sequence<LP - 1ull>(), make_index_sequence<LF - 1ull>(), make_index_sequence<LE>())...
		);
	}
}

#endif//_STP_FILE_H_