//INLINE DEFINITION FOR HASH COMBINE, DO NOT INCLUDE MANUALLY

#ifdef _STP_HASH_COMBINE_H_

template<typename T>
inline void SuperTerrainPlus::STPHashCombine::STPHashCombineImpl::combineOne(size_t& seed, const T& value) {
	//The algorithm is based on boost::hash_combine
	seed ^= std::hash<T>()(value) + 0x9e3779b9ull + (seed << 6ull) + (seed >> 2ull);
}

template<typename ...T>
inline size_t SuperTerrainPlus::STPHashCombine::combine(size_t seed, const T&... value) {
	(STPHashCombineImpl::combineOne(seed, value), ...);
	return seed;
}

#endif//_STP_HASH_COMBINE_H_