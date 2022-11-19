//INLINE TEMPLATE DEFINITION FOR NULLABLE PRIMITIVE

#ifdef _STP_NULLABLE_PRIMITIVE_H_

template<typename Pri, Pri Null>
template<class Del>
inline void SuperTerrainPlus::STPNullablePrimitive<Pri, Null>::STPNullableDeleter<Del>::operator()(const pointer ptr) const noexcept {
	this->Deleter(ptr);
}

template<typename Pri, Pri Null>
inline SuperTerrainPlus::STPNullablePrimitive<Pri, Null>::STPNullablePrimitive(std::nullptr_t) noexcept : Value(Null) {

}

template<typename Pri, Pri Null>
inline SuperTerrainPlus::STPNullablePrimitive<Pri, Null>::STPNullablePrimitive(const Pri value) noexcept : Value(value) {

}

template<typename Pri, Pri Null>
inline SuperTerrainPlus::STPNullablePrimitive<Pri, Null>& SuperTerrainPlus::STPNullablePrimitive<Pri, Null>::operator=(
	std::nullptr_t) noexcept {
	this->Value = Null;
	return *this;
}

template<typename Pri, Pri Null>
inline SuperTerrainPlus::STPNullablePrimitive<Pri, Null>::operator Pri() const noexcept {
	return this->Value;
}

template<typename Pri, Pri Null>
inline SuperTerrainPlus::STPNullablePrimitive<Pri, Null>::operator bool() const noexcept {
	return this->Value != Null;
}

template<typename Pri, Pri Null>
inline bool SuperTerrainPlus::STPNullablePrimitive<Pri, Null>::operator==(const STPNullablePrimitive p) const noexcept {
	return this->Value == p.Value;
}

template<typename Pri, Pri Null>
inline bool SuperTerrainPlus::STPNullablePrimitive<Pri, Null>::operator!=(const STPNullablePrimitive p) const noexcept {
	return this->Value != p.Value;
}

template<typename Pri, Pri Null>
inline bool SuperTerrainPlus::STPNullablePrimitive<Pri, Null>::operator==(std::nullptr_t) const noexcept {
	return this->Value == Null;
}

template<typename Pri, Pri Null>
inline bool SuperTerrainPlus::STPNullablePrimitive<Pri, Null>::operator!=(std::nullptr_t) const noexcept {
	return this->Value != Null;
}

#endif//_STP_NULLABLE_PRIMITIVE_H_