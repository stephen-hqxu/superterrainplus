#include <SuperRealism+/Object/STPSampler.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/gtc/type_ptr.hpp>

using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

inline static GLuint createSampler() noexcept {
	GLuint sampler;
	glCreateSamplers(1u, &sampler);
	return sampler;
}

void STPSampler::STPSamplerUnbinder::operator()(STPOpenGL::STPuint unit) const noexcept {
	glBindSampler(unit, 0u);
}

void STPSampler::STPSamplerDeleter::operator()(STPOpenGL::STPuint sampler) const noexcept {
	glDeleteSamplers(1u, &sampler);
}

STPSampler::STPSampler() noexcept : Sampler(createSampler()) {

}

SuperTerrainPlus::STPOpenGL::STPuint STPSampler::operator*() const noexcept {
	return this->Sampler.get();
}

void STPSampler::filter(STPOpenGL::STPint min, STPOpenGL::STPint mag) noexcept {
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_MIN_FILTER, min);
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_MAG_FILTER, mag);
}

void STPSampler::wrap(STPOpenGL::STPint s, STPOpenGL::STPint t, STPOpenGL::STPint r) noexcept {
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_WRAP_S, s);
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_WRAP_T, t);
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_WRAP_R, r);
}

void STPSampler::wrap(STPOpenGL::STPint str) noexcept {
	this->wrap(str, str, str);
}

void STPSampler::borderColor(STPGLVector::STPfloatVec4 color) noexcept {
	glSamplerParameterfv(this->Sampler.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPSampler::borderColor(STPGLVector::STPintVec4 color) noexcept {
	glSamplerParameterIiv(this->Sampler.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPSampler::borderColor(STPGLVector::STPuintVec4 color) noexcept {
	glSamplerParameterIuiv(this->Sampler.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPSampler::anisotropy(STPOpenGL::STPfloat ani) noexcept {
	glSamplerParameterf(this->Sampler.get(), GL_TEXTURE_MAX_ANISOTROPY, ani);
}

void STPSampler::compareFunction(STPOpenGL::STPint function) noexcept {
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_COMPARE_FUNC, function);
}

void STPSampler::compareMode(STPOpenGL::STPint mode) noexcept {
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_COMPARE_MODE, mode);
}

STPSampler::STPSamplerUnitStateManager STPSampler::bindManaged(STPOpenGL::STPuint unit) const noexcept {
	glBindSampler(unit, this->Sampler.get());
	return STPSamplerUnitStateManager(unit);
}