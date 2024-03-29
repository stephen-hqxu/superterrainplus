#ifdef _STP_MATRIX_4X4D_H_

//In documentation, matrix is assumed to have this layout.
//GLM uses column-major, so this is in column major as well, although I prefer row major...
/*
[ 0 4  8 12 ]
[ 1 5  9 13 ]
[ 2 6 10 14 ]
[ 3 7 11 15 ]
*/

#define FOREACH_COL_BEG() for (int i = 0; i < 4; i++) {
#define FOREACH_COL_END() }

inline SuperTerrainPlus::STPMatrix4x4d::STPMatrix4x4d(const glm::dmat4 & mat) noexcept {
	FOREACH_COL_BEG()
		this->Mat[i] = STPVector4d(mat[i]);
	FOREACH_COL_END()
}

inline const __m256d& SuperTerrainPlus::STPMatrix4x4d::get(const size_t idx) const noexcept {
	return this->Mat[idx].Vec;
}

inline __m256d& SuperTerrainPlus::STPMatrix4x4d::get(const size_t idx) noexcept {
	return const_cast<__m256d&>(const_cast<const STPMatrix4x4d*>(this)->get(idx));
}

inline SuperTerrainPlus::STPMatrix4x4d::operator glm::dmat4() const noexcept {
	glm::dmat4 res;
	FOREACH_COL_BEG()
		res[i] = static_cast<glm::dvec4>(this->Mat[i]);
	FOREACH_COL_END()
	return res;
}

inline SuperTerrainPlus::STPMatrix4x4d::operator glm::mat4() const noexcept {
	glm::mat4 res;
	FOREACH_COL_BEG()
		res[i] = static_cast<glm::vec4>(this->Mat[i]);
	FOREACH_COL_END()
	return res;
}

inline const SuperTerrainPlus::STPVector4d& SuperTerrainPlus::STPMatrix4x4d::operator[](const size_t idx) const noexcept {
	return this->Mat[idx];
}

inline SuperTerrainPlus::STPVector4d& SuperTerrainPlus::STPMatrix4x4d::operator[](const size_t idx) noexcept {
	return const_cast<STPVector4d&>(const_cast<const STPMatrix4x4d*>(this)->operator[](idx));
}

inline SuperTerrainPlus::STPMatrix4x4d SuperTerrainPlus::STPMatrix4x4d::operator*(const STPMatrix4x4d& rhs) const noexcept {
	STPMatrix4x4d m;
	FOREACH_COL_BEG()
		m[i] = (*this) * rhs[i];
	FOREACH_COL_END()
	return m;
}

inline SuperTerrainPlus::STPVector4d SuperTerrainPlus::STPMatrix4x4d::operator*(const STPVector4d& rhs) const noexcept {
	const __m256d& v = rhs.Vec;
		//0 0 0 0
	const __m256d s0 = _mm256_permute4x64_pd(v, STP_MM_BIT8(0, 0, 0, 0)),
		//1 1 1 1
		s1 = _mm256_permute4x64_pd(v, STP_MM_BIT8(1, 1, 1, 1)),
		//2 2 2 2
		s2 = _mm256_permute4x64_pd(v, STP_MM_BIT8(2, 2, 2, 2)),
		//3 3 3 3
		s3 = _mm256_permute4x64_pd(v, STP_MM_BIT8(3, 3, 3, 3));

	const __m256d t0 = _mm256_fmadd_pd(this->get(0), s0, _mm256_mul_pd(this->get(1), s1)),
		t1 = _mm256_fmadd_pd(this->get(2), s2, _mm256_mul_pd(this->get(3), s3));

	return _mm256_add_pd(t0, t1);
}

inline SuperTerrainPlus::STPMatrix4x4d SuperTerrainPlus::STPMatrix4x4d::transpose() const noexcept {
		//0 4 2 6
	const __m256d s0 = _mm256_shuffle_pd(this->get(0), this->get(1), STP_MM_BIT4(0, 0, 0, 0)),
		//1 5 3 7
		s1 = _mm256_shuffle_pd(this->get(0), this->get(1), STP_MM_BIT4(1, 1, 1, 1)),
		//8 12 10 14
		s2 = _mm256_shuffle_pd(this->get(2), this->get(3), STP_MM_BIT4(0, 0, 0, 0)),
		//9 13 11 15
		s3 = _mm256_shuffle_pd(this->get(2), this->get(3), STP_MM_BIT4(1, 1, 1, 1));

	STPMatrix4x4d m;
	//0 4 8 12
	m.get(0) = _mm256_permute2f128_pd(s0, s2, STP_MM_BIT8(0, 0, 2, 0));
	//2 6 10 14
	m.get(2) = _mm256_permute2f128_pd(s0, s2, STP_MM_BIT8(1, 0, 3, 0));
	//1 5 9 13
	m.get(1) = _mm256_permute2f128_pd(s1, s3, STP_MM_BIT8(0, 0, 2, 0));
	//3 7 11 15
	m.get(3) = _mm256_permute2f128_pd(s1, s3, STP_MM_BIT8(1, 0, 3, 0));

	return m;
}

inline SuperTerrainPlus::STPMatrix4x4d SuperTerrainPlus::STPMatrix4x4d::inverse() const noexcept {
	const static __m256d SignA = _mm256_set_pd(1.0, -1.0, 1.0, -1.0),
		SignB = _mm256_set_pd(-1.0, 1.0, -1.0, 1.0),
		One = _mm256_set1_pd(1.0);

		//12 8 14 10
	const __m256d t00 = _mm256_shuffle_pd(this->get(3), this->get(2), STP_MM_BIT4(0, 0, 0, 0)),
		//13 9 5 11
		t01 = _mm256_shuffle_pd(this->get(3), this->get(2), STP_MM_BIT4(1, 1, 1, 1)),
		//8 4 10 6
		t10 = _mm256_shuffle_pd(this->get(2), this->get(1), STP_MM_BIT4(0, 0, 0, 0)),
		//9 5 11 7
		t11 = _mm256_shuffle_pd(this->get(2), this->get(1), STP_MM_BIT4(1, 1, 1, 1)),
		//4 0 6 2
		t20 = _mm256_shuffle_pd(this->get(1), this->get(0), STP_MM_BIT4(0, 0, 0, 0)),
		//5 1 7 3
		t21 = _mm256_shuffle_pd(this->get(1), this->get(0), STP_MM_BIT4(1, 1, 1, 1));

		//14 14 14 10
	const __m256d r00 = _mm256_permute4x64_pd(t00, STP_MM_BIT8(2, 2, 2, 3)),
		//15 15 15 11
		r01 = _mm256_permute4x64_pd(t01, STP_MM_BIT8(2, 2, 2, 3)),
		//12 12 12 8
		r02 = _mm256_permute4x64_pd(t00, STP_MM_BIT8(0, 0, 0, 1)),
		//13 13 13 9
		r03 = _mm256_permute4x64_pd(t01, STP_MM_BIT8(0, 0, 0, 1)),
		//10 10 6 6
		r10 = _mm256_permute4x64_pd(t10, STP_MM_BIT8(2, 2, 3, 3)),
		//11 11 7 7
		r11 = _mm256_permute4x64_pd(t11, STP_MM_BIT8(2, 2, 3, 3)),
		//8 8 4 4
		r12 = _mm256_permute4x64_pd(t10, STP_MM_BIT8(0, 0, 1, 1)),
		//9 9 5 5
		r13 = _mm256_permute4x64_pd(t11, STP_MM_BIT8(0, 0, 1, 1)),
		//Vec0: 4 0 0 0
		v0 = _mm256_permute4x64_pd(t20, STP_MM_BIT8(0, 1, 1, 1)),
		//Vec1: 5 1 1 1
		v1 = _mm256_permute4x64_pd(t21, STP_MM_BIT8(0, 1, 1, 1)),
		//Vec2: 6 2 2 2
		v2 = _mm256_permute4x64_pd(t20, STP_MM_BIT8(2, 3, 3, 3)),
		//Vec3: 7 3 3 3
		v3 = _mm256_permute4x64_pd(t21, STP_MM_BIT8(2, 3, 3, 3));
		
		//factor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		//factor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		//factor06 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
		//factor13 = m[1][2] * m[2][3] - m[2][2] * m[1][3];
	const __m256d fac0 = _mm256_fmsub_pd(r10, r01, _mm256_mul_pd(r00, r11)),
		//factor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		//factor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		//factor07 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
		//factor14 = m[1][1] * m[2][3] - m[2][1] * m[1][3];
		fac1 = _mm256_fmsub_pd(r13, r01, _mm256_mul_pd(r03, r11)),
		//factor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		//factor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		//factor08 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
		//factor15 = m[1][1] * m[2][2] - m[2][1] * m[1][2];
		fac2 = _mm256_fmsub_pd(r13, r00, _mm256_mul_pd(r03, r10)),
		//factor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		//factor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		//factor09 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
		//factor16 = m[1][0] * m[2][3] - m[2][0] * m[1][3];
		fac3 = _mm256_fmsub_pd(r12, r01, _mm256_mul_pd(r02, r11)),
		//factor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		//factor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		//factor10 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
		//factor17 = m[1][0] * m[2][2] - m[2][0] * m[1][2];
		fac4 = _mm256_fmsub_pd(r12, r00, _mm256_mul_pd(r02, r10)),
		//factor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
		//factor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
		//factor12 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
		//factor18 = m[1][0] * m[2][1] - m[2][0] * m[1][1];
		fac5 = _mm256_fmsub_pd(r12, r03, _mm256_mul_pd(r02, r13));

		//col0
		//+ (Vec1[0] * Fac0[0] - Vec2[0] * Fac1[0] + Vec3[0] * Fac2[0]),
		//- (Vec1[1] * Fac0[1] - Vec2[1] * Fac1[1] + Vec3[1] * Fac2[1]),
		//+ (Vec1[2] * Fac0[2] - Vec2[2] * Fac1[2] + Vec3[2] * Fac2[2]),
		//- (Vec1[3] * Fac0[3] - Vec2[3] * Fac1[3] + Vec3[3] * Fac2[3]),
	const __m256d inv0 = _mm256_mul_pd(
		SignB,
		_mm256_fmadd_pd(v3, fac2,
			_mm256_fmsub_pd(v1, fac0,
				_mm256_mul_pd(v2, fac1)
			)
		)
		//col1
		//- (Vec0[0] * Fac0[0] - Vec2[0] * Fac3[0] + Vec3[0] * Fac4[0]),
		//+ (Vec0[0] * Fac0[1] - Vec2[1] * Fac3[1] + Vec3[1] * Fac4[1]),
		//- (Vec0[0] * Fac0[2] - Vec2[2] * Fac3[2] + Vec3[2] * Fac4[2]),
		//+ (Vec0[0] * Fac0[3] - Vec2[3] * Fac3[3] + Vec3[3] * Fac4[3]),
	), inv1 = _mm256_mul_pd(
		SignA,
		_mm256_fmadd_pd(v3, fac4,
			_mm256_fmsub_pd(v0, fac0,
				_mm256_mul_pd(v2, fac3)
			)
		)
		//col2
		//+ (Vec0[0] * Fac1[0] - Vec1[0] * Fac3[0] + Vec3[0] * Fac5[0]),
		//- (Vec0[0] * Fac1[1] - Vec1[1] * Fac3[1] + Vec3[1] * Fac5[1]),
		//+ (Vec0[0] * Fac1[2] - Vec1[2] * Fac3[2] + Vec3[2] * Fac5[2]),
		//- (Vec0[0] * Fac1[3] - Vec1[3] * Fac3[3] + Vec3[3] * Fac5[3]),
	), inv2 = _mm256_mul_pd(
		SignB,
		_mm256_fmadd_pd(v3, fac5,
			_mm256_fmsub_pd(v0, fac1,
				_mm256_mul_pd(v1, fac3)
			)
		)
		//col3
		//- (Vec1[0] * Fac2[0] - Vec1[0] * Fac4[0] + Vec2[0] * Fac5[0]),
		//+ (Vec1[0] * Fac2[1] - Vec1[1] * Fac4[1] + Vec2[1] * Fac5[1]),
		//- (Vec1[0] * Fac2[2] - Vec1[2] * Fac4[2] + Vec2[2] * Fac5[2]),
		//+ (Vec1[0] * Fac2[3] - Vec1[3] * Fac4[3] + Vec2[3] * Fac5[3]));
	), inv3 = _mm256_mul_pd(
		SignA,
		_mm256_fmadd_pd(v2, fac5,
			_mm256_fmsub_pd(v0, fac2,
				_mm256_mul_pd(v1, fac4)
			)
		)
	);
		
		//0 4 2 6
	const __m256d s0 = _mm256_shuffle_pd(inv0, inv1, STP_MM_BIT4(0, 0, 0, 0)),
		//8 12 10 14
		s1 = _mm256_shuffle_pd(inv2, inv3, STP_MM_BIT4(0, 0, 0, 0));
		//0 4 8 12
	const __m256d col = _mm256_permute2f128_pd(s0, s1, STP_MM_BIT8(0, 0, 2, 0));

		//Determinant = m[0][0] * Inverse[0][0]
		//				+ m[0][1] * Inverse[1][0]
		//				+ m[0][2] * Inverse[2][0]
		//				+ m[0][3] * Inverse[3][0];
	const __m256d det = STPVector4d::dotVector4dRaw(this->get(0), col),
		det_rcp = _mm256_div_pd(One, det);

	STPMatrix4x4d m;
	m.get(0) = _mm256_mul_pd(inv0, det_rcp);
	m.get(1) = _mm256_mul_pd(inv1, det_rcp);
	m.get(2) = _mm256_mul_pd(inv2, det_rcp);
	m.get(3) = _mm256_mul_pd(inv3, det_rcp);
	return m;
}

inline SuperTerrainPlus::STPMatrix4x4d::STPMatrix3x3d SuperTerrainPlus::STPMatrix4x4d::asMatrix3x3d() const noexcept {
	const static __m256d Factor = _mm256_set_pd(0.0, 1.0, 1.0, 1.0),
		Preserver = _mm256_set_pd(1.0, 0.0, 0.0, 0.0);
	//Basically we want something like this:
	/*
	[ a b c 0 ]
	[ d e f 0 ]
	[ g h i 0 ]
	[ 0 0 0 1 ]
	*/
	STPMatrix3x3d m;
	for (int i = 0; i < 3; i++) {
		m.get(i) = _mm256_blend_pd(this->get(i), Factor, STP_MM_BIT4(0, 0, 0, 1));
	}
	m.get(3) = Preserver;
	return m;
}

#undef FOREACH_COL_BEG
#undef FOREACH_COL_END

#endif//_STP_MATRIX_4X4D_H_