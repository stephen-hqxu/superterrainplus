#ifdef _STP_VECTOR_4D_H_

#include <glm/gtc/type_ptr.hpp>

inline __m256d SuperTerrainPlus::STPVector4d::dotVector4dRaw(const __m256d& lhs, const __m256d& rhs) noexcept {
	const __m256d mul = _mm256_mul_pd(lhs, rhs),
		//double horizontal add does not cross the lane, need to shuffle it
		h = _mm256_permute4x64_pd(_mm256_hadd_pd(mul, mul), STP_MM_BIT8(0, 2, 1, 3));
	return _mm256_hadd_pd(h, h);
}

inline SuperTerrainPlus::STPVector4d::STPVector4d(const __m256d& vec) noexcept : Vec(vec) {

}

inline SuperTerrainPlus::STPVector4d::STPVector4d() noexcept : STPVector4d(_mm256_setzero_pd()) {

}

inline SuperTerrainPlus::STPVector4d::STPVector4d(const glm::dvec4& vec) noexcept {
	const double* const vec_addr = glm::value_ptr(vec);
	if (STPAlgebraCommon::isAVXAligned(vec_addr)) {
		//if aligned, use the faster load instruction
		this->Vec = _mm256_load_pd(vec_addr);
		return;
	}
	this->Vec = _mm256_loadu_pd(vec_addr);
}

inline SuperTerrainPlus::STPVector4d::operator glm::dvec4() const noexcept {
	alignas(STPAlgebraCommon::AVXAlignment) glm::dvec4 res;
	_mm256_store_pd(glm::value_ptr(res), this->Vec);
	return res;
}

inline SuperTerrainPlus::STPVector4d::operator glm::vec4() const noexcept {
	alignas(STPAlgebraCommon::SSEAlignment) glm::vec4 res;
	_mm_store_ps(glm::value_ptr(res), _mm256_cvtpd_ps(this->Vec));
	return res;
}

inline SuperTerrainPlus::STPVector4d SuperTerrainPlus::STPVector4d::operator+(const STPVector4d& rhs) const noexcept {
	return _mm256_add_pd(this->Vec, rhs.Vec);
}

inline SuperTerrainPlus::STPVector4d SuperTerrainPlus::STPVector4d::operator/(const STPVector4d& rhs) const noexcept {
	return _mm256_div_pd(this->Vec, rhs.Vec);
}

template<SuperTerrainPlus::STPVector4d::STPElement E>
inline SuperTerrainPlus::STPVector4d SuperTerrainPlus::STPVector4d::broadcast() const noexcept {
	constexpr static auto i = static_cast<std::underlying_type_t<STPElement>>(E);
	return _mm256_permute4x64_pd(this->Vec, STP_MM_BIT8(i, i, i, i));
}

inline double SuperTerrainPlus::STPVector4d::dot(const SuperTerrainPlus::STPVector4d& rhs) const noexcept {
	const __m256d vec_dot = STPVector4d::dotVector4dRaw(this->Vec, rhs.Vec);
	//all components have the same value, extract any one of them
	return _mm256_cvtsd_f64(vec_dot);
}

#endif//_STP_VECTOR_4D_H_