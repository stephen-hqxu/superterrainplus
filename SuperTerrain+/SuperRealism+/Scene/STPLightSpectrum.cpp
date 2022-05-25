#include <SuperRealism+/Scene/Light/STPLightSpectrum.h>
//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/vec2.hpp>
#include <glm/gtc/type_ptr.hpp>

using glm::uvec2;
using glm::ivec3;
using glm::uvec3;
using glm::vec3;

using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

STPLightSpectrum::STPLightSpectrum(unsigned int length, STPOpenGL::STPenum format) : Spectrum(GL_TEXTURE_1D), SpectrumLength(length) {
	//check for argument
	if (this->SpectrumLength == 0u) {
		throw STPException::STPBadNumericRange("The length of the spectrum should be a positive integer");
	}

	//setup output texture
	this->Spectrum.textureStorage<STPTexture::STPDimension::ONE>(1, format, uvec3(this->SpectrumLength, uvec2(1u)));

	this->Spectrum.filter(GL_NEAREST, GL_LINEAR);
	this->Spectrum.wrap(GL_CLAMP_TO_EDGE);

	//create spectrum texture handle
	this->SpectrumHandle = STPBindlessTexture(this->Spectrum);
}

const STPTexture& STPLightSpectrum::spectrum() const {
	return this->Spectrum;
}

SuperTerrainPlus::STPOpenGL::STPuint64 STPLightSpectrum::spectrumHandle() const {
	return *this->SpectrumHandle;
}

void STPLightSpectrum::setData(const STPColorArray& color) {
	if (color.size() > this->SpectrumLength) {
		throw STPException::STPMemoryError("There is insufficient amount of memory to hold all colours specified in the array");
	}

	this->Spectrum.textureSubImage<STPTexture::STPDimension::ONE>(0, ivec3(0), uvec3(color.size(), uvec2(1u)), GL_RGB, GL_FLOAT, color.data());
}