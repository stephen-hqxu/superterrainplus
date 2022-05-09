#include <SuperTerrain+/Utility/STPNullablePrimitive.h>

//GLAD
#include <glad/glad.h>

using namespace SuperTerrainPlus;

template<class Pri, Pri Null>
STPNullablePrimitive<Pri, Null>::STPNullablePrimitive(std::nullptr_t) : Value(Null) {

}

template<class Pri, Pri Null>
STPNullablePrimitive<Pri, Null>::STPNullablePrimitive(Pri value) : Value(value) {

}

template<class Pri, Pri Null>
STPNullablePrimitive<Pri, Null>::operator Pri() const {
	return this->Value;
}

template<class Pri, Pri Null>
Pri STPNullablePrimitive<Pri, Null>::operator*() const {
	return this->Value;
}

template<class Pri, Pri Null>
bool STPNullablePrimitive<Pri, Null>::operator==(std::nullptr_t) const {
	return this->Value == Null;
}

template<class Pri, Pri Null>
bool STPNullablePrimitive<Pri, Null>::operator!=(std::nullptr_t) const {
	return this->Value != Null;
}

//Explicit instantiation
template struct STP_API STPNullablePrimitive<GLuint, 0u>;
template struct STP_API STPNullablePrimitive<GLuint64, 0ull>;