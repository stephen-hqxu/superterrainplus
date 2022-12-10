#include <SuperRealism+/Scene/Light/STPLightSpectrum.h>
//Error
#include <SuperTerrain+/Exception/STPNumericDomainError.h>

//GLAD
#include <glad/glad.h>

using namespace SuperTerrainPlus::STPRealism;

STPLightSpectrum::STPLightSpectrum(const unsigned int length, const STPOpenGL::STPenum format) : Spectrum(GL_TEXTURE_1D), SpectrumLength(length) {
	//check for argument
	STP_ASSERTION_NUMERIC_DOMAIN(this->SpectrumLength > 0u, "The length of the spectrum should be a positive integer");

	//setup output texture
	this->Spectrum.textureStorage1D(1, format, this->SpectrumLength);

	this->Spectrum.filter(GL_NEAREST, GL_LINEAR);
	this->Spectrum.wrap(GL_CLAMP_TO_EDGE);

	//create spectrum texture handle
	this->SpectrumHandle = STPBindlessTexture::make(this->Spectrum);
}

const STPTexture& STPLightSpectrum::spectrum() const noexcept {
	return this->Spectrum;
}

SuperTerrainPlus::STPOpenGL::STPuint64 STPLightSpectrum::spectrumHandle() const noexcept {
	return this->SpectrumHandle.get();
}

void STPLightSpectrum::setData(const STPOpenGL::STPsizei size, const STPOpenGL::STPenum format,
	const STPOpenGL::STPenum type, const void* const data) noexcept {
	this->Spectrum.textureSubImage1D(0, 0, size, format, type, data);
}