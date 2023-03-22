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

void STPSampler::STPSamplerUnbinder::operator()(const STPOpenGL::STPuint unit) const noexcept {
	glBindSampler(unit, 0u);
}

void STPSampler::STPSamplerDeleter::operator()(const STPOpenGL::STPuint sampler) const noexcept {
	glDeleteSamplers(1u, &sampler);
}

STPSampler::STPSampler() noexcept : Sampler(createSampler()) {

}

SuperTerrainPlus::STPOpenGL::STPuint STPSampler::operator*() const noexcept {
	return this->Sampler.get();
}

void STPSampler::filter(const STPOpenGL::STPint min, const STPOpenGL::STPint mag) noexcept {
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_MIN_FILTER, min);
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_MAG_FILTER, mag);
}

void STPSampler::wrap(const STPOpenGL::STPint s, const STPOpenGL::STPint t, const STPOpenGL::STPint r) noexcept {
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_WRAP_S, s);
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_WRAP_T, t);
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_WRAP_R, r);
}

void STPSampler::wrap(const STPOpenGL::STPint str) noexcept {
	this->wrap(str, str, str);
}

void STPSampler::borderColor(const STPGLVector::STPfloatVec4 color) noexcept {
	glSamplerParameterfv(this->Sampler.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPSampler::borderColor(const STPGLVector::STPintVec4 color) noexcept {
	glSamplerParameterIiv(this->Sampler.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPSampler::borderColor(const STPGLVector::STPuintVec4 color) noexcept {
	glSamplerParameterIuiv(this->Sampler.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPSampler::anisotropy(const STPOpenGL::STPfloat ani) noexcept {
	glSamplerParameterf(this->Sampler.get(), GL_TEXTURE_MAX_ANISOTROPY, ani);
}

void STPSampler::compareFunction(const STPOpenGL::STPint function) noexcept {
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_COMPARE_FUNC, function);
}

void STPSampler::compareMode(const STPOpenGL::STPint mode) noexcept {
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_COMPARE_MODE, mode);
}

STPSampler::STPSamplerUnitStateManager STPSampler::bindManaged(const STPOpenGL::STPuint unit) const noexcept {
	glBindSampler(unit, this->Sampler.get());
	return STPSamplerUnitStateManager(unit);
}