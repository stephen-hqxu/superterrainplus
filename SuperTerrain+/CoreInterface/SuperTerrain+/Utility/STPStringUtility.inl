#ifdef _STP_STRING_UTILITY_H_

template<size_t... L>
inline constexpr auto SuperTerrainPlus::STPStringUtility::concatCharArray(const char (&... str)[L]) noexcept {
	constexpr size_t totalLength = (0u + ... + L);
	//remember to add a null to the end of the string
	std::array<char, totalLength + 1u> arr = { };

	//TODO: we can use std::for_each in C++20 because it is constexpr
	auto append = [arrIdx = size_t(0u), &arr](const char* carr, size_t len) mutable constexpr -> void {
		for (size_t i = 0u; i < len; i++) {
			arr[arrIdx++] = carr[i];
		}
	};
	//remember to remove the last character from each string
	(append(str, L - 1u), ...);
	//add null
	arr[totalLength] = 0;

	return arr;
}

template<size_t LP, size_t LF, size_t... LE>
inline constexpr auto SuperTerrainPlus::STPStringUtility::generateFilename(const char (&path)[LP], const char (&file)[LF], const char (&... extension)[LE]) noexcept {
	constexpr auto allSameSize = [](const auto& x, const auto&... xs) constexpr -> bool {
		return ((x == xs) && ... && true);
	};

	if constexpr (allSameSize(LE...)) {
		//all filenames have the same size, a simple array can be used
		//each string ends with a null, so we need to eliminate the null symbol for all strings except the last one
		return std::array { STPStringUtility::concatCharArray(path, file, extension)... };
	} else {
		//to hold elements of different types, use tuple
		return std::make_tuple(STPStringUtility::concatCharArray(path, file, extension)...);
	}
}

#endif//_STP_STRING_UTILITY_H_