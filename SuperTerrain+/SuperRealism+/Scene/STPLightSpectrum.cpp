#include <SuperRealism+/Scene/Light/STPLightSpectrum.h>
//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/vec2.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <type_traits>

using glm::u8vec3;
using glm::vec3;

using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

STPLightSpectrum::STPLightSpectrum(unsigned int length, STPOpenGL::STPenum format) : Spectrum(GL_TEXTURE_1D), SpectrumLength(length) {
	//check for argument
	if (this->SpectrumLength == 0u) {
		throw STPException::STPBadNumericRange("The length of the spectrum should be a positive integer");
	}

	//setup output texture
	this->Spectrum.textureStorage1D(1, format, this->SpectrumLength);

	this->Spectrum.filter(GL_NEAREST, GL_LINEAR);
	this->Spectrum.wrap(GL_CLAMP_TO_EDGE);

	//create spectrum texture handle
	this->SpectrumHandle = STPBindlessTexture::make(this->Spectrum);
}

const STPTexture& STPLightSpectrum::spectrum() const {
	return this->Spectrum;
}

SuperTerrainPlus::STPOpenGL::STPuint64 STPLightSpectrum::spectrumHandle() const {
	return this->SpectrumHandle.get();
}

template<typename T>
void STPLightSpectrum::setData(const STPColourArray<T>& color) {
	if (color.size() > this->SpectrumLength) {
		throw STPException::STPMemoryError("There is insufficient amount of memory to hold all colours specified in the array");
	}

	using std::is_same_v;
	//determine format and type
	GLenum format = 0u, type = 0u;
	if constexpr (is_same_v<T, vec3>) {
		format = GL_RGB;
		type = GL_FLOAT;
	} else if constexpr (is_same_v<T, u8vec3>) {
		format = GL_RGB;
		type = GL_UNSIGNED_BYTE;
	}

	this->Spectrum.textureSubImage1D(0, 0, static_cast<STPOpenGL::STPsizei>(color.size()), format, type, color.data());
}

//Explicit Instantiation
#define SET_DATA(TYPE) template STP_REALISM_API void STPLightSpectrum::setData<TYPE>(const STPColourArray<TYPE>&)
SET_DATA(vec3);
SET_DATA(u8vec3);