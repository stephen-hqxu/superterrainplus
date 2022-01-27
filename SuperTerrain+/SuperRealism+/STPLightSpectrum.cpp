#include <SuperRealism+/Scene/STPLightSpectrum.h>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>

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

STPLightSpectrum::STPLightSpectrum(unsigned int length, STPSpectrumType type, STPOpenGL::STPenum format) : 
	Spectrum((type == STPSpectrumType::Monotonic) ? GL_TEXTURE_1D : GL_TEXTURE_1D_ARRAY), SpectrumLength(length) {
	//check for argument
	if (this->SpectrumLength == 0u) {
		throw STPException::STPBadNumericRange("The length of the spectrum should be a positive integer");
	}

	//setup output texture
	if (type == STPSpectrumType::Monotonic) {
		this->Spectrum.textureStorage<STPTexture::STPDimension::ONE>(1, format, uvec3(this->SpectrumLength, uvec2(1u)));
	}
	else {
		this->Spectrum.textureStorage<STPTexture::STPDimension::TWO>(1, format, uvec3(this->SpectrumLength, 2u, 1u));
	}
	this->Spectrum.filter(GL_NEAREST, GL_LINEAR);
	this->Spectrum.wrap(GL_CLAMP_TO_EDGE);
}

const STPTexture& STPLightSpectrum::spectrum() const {
	return this->Spectrum;
}

STPStaticLightSpectrum::STPStaticLightSpectrum() : STPLightSpectrum(1u, STPSpectrumType::Monotonic, GL_RGB8) {

}

void STPStaticLightSpectrum::operator()(vec3 color) {
	this->Spectrum.textureSubImage<STPTexture::STPDimension::ONE>(0, ivec3(0), uvec3(1u), GL_RGB, GL_FLOAT, value_ptr(color));
}

float STPStaticLightSpectrum::coordinate() const {
	return 0.0f;
}

STPArrayLightSpectrum::STPArrayLightSpectrum(unsigned int length) : STPLightSpectrum(length, STPSpectrumType::Monotonic, GL_RGB8), SampleCoordinate(0.0f) {

}

void STPArrayLightSpectrum::operator()(const STPColorArray& color) {
	this->Spectrum.textureSubImage<STPTexture::STPDimension::ONE>(0, ivec3(0), uvec3(color.size(), uvec2(1u)), GL_RGB, GL_FLOAT, color.data());
}

void STPArrayLightSpectrum::setCoordinate(float coord) {
	this->SampleCoordinate = coord;
}

float STPArrayLightSpectrum::coordinate() const {
	return this->SampleCoordinate;
}