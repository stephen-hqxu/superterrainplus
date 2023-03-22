#ifdef _STP_ALGEBRA_COMMON_H_

inline bool SuperTerrainPlus::STPAlgebraCommon::isAVXAligned(const void* const addr) noexcept {
	return !(reinterpret_cast<std::uintptr_t>(addr) & (AVXAlignment - 1ull));
}

#endif//_STP_ALGEBRA_COMMON_H_