//INLINE DEFINITION FOR HASH COMBINE, DO NOT INCLUDE MANUALLY

#ifdef _STP_HASH_COMBINE_H_

template<typename T>
inline void SuperTerrainPlus::STPHashCombine::combine(size_t& seed, T value) {
	std::hash<T> hasher;
	
	//The algorithm is based on boost::hash_combine
	seed ^= hasher(value) + 0x9e3779b9ull + (seed << 6ull) + (seed >> 2ull);
}

template<typename ...T>
inline void SuperTerrainPlus::STPHashCombine::combineAll(size_t& seed, T... value) {
	(STPHashCombine::combine(seed, value), ...);
}

#endif//_STP_HASH_COMBINE_H_