#include <SuperRealism+/Renderer/STPLightSpectrum.h>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>

using glm::ivec3;
using glm::uvec3;
using glm::vec4;
using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

STPLightSpectrum::STPLightSpectrum(unsigned int length) : Spectrum(GL_TEXTURE_1D_ARRAY), SpectrumLength(length) {
	//check for argument
	if (this->SpectrumLength == 0u) {
		throw STPException::STPBadNumericRange("The length of the spectrum should be a positive integer");
	}

	//setup output texture
	this->Spectrum.textureStorage<STPTexture::STPDimension::TWO>(1, GL_RGBA16F, uvec3(this->SpectrumLength, 2u, 1u));
	this->Spectrum.filter(GL_NEAREST, GL_LINEAR);
	this->Spectrum.wrap(GL_CLAMP_TO_EDGE);
}

const STPTexture& STPLightSpectrum::spectrum() const {
	return this->Spectrum;
}

STPStaticLightSpectrum::STPStaticLightSpectrum() : STPLightSpectrum(1u) {

}

void STPStaticLightSpectrum::operator()(const vec4& indirect, const vec4& direct) {
	this->Spectrum.textureSubImage<STPTexture::STPDimension::TWO>(0, ivec3(0), uvec3(1u), GL_RGBA, GL_FLOAT, value_ptr(indirect));
	this->Spectrum.textureSubImage<STPTexture::STPDimension::TWO>(0, ivec3(0, 1, 0), uvec3(1u), GL_RGBA, GL_FLOAT, value_ptr(direct));
}

float STPStaticLightSpectrum::coordinate() const {
	return 0.0f;
}

STPArrayLightSpectrum::STPArrayLightSpectrum(unsigned int length) : STPLightSpectrum(length), SampleCoordinate(0.0f) {

}

void STPArrayLightSpectrum::operator()(const STPColorArray& indirect, const STPColorArray& direct) {
	if (indirect.size() != direct.size()) {
		throw STPException::STPMemoryError("Indirect light spectrum has different size from direct light spectrum");
	}

	this->Spectrum.textureSubImage<STPTexture::STPDimension::TWO>(0, ivec3(0), uvec3(indirect.size(), 1u, 1u), GL_RGBA, GL_FLOAT, indirect.data());
	this->Spectrum.textureSubImage<STPTexture::STPDimension::TWO>(0, ivec3(0, 1, 0), uvec3(direct.size(), 1u, 1u), GL_RGBA, GL_FLOAT, direct.data());
}

void STPArrayLightSpectrum::setCoordinate(float coord) {
	this->SampleCoordinate = coord;
}

float STPArrayLightSpectrum::coordinate() const {
	return this->SampleCoordinate;
}