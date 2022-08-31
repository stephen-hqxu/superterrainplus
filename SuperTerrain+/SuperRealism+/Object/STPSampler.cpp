#include <SuperRealism+/Object/STPSampler.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/gtc/type_ptr.hpp>

using glm::ivec4;
using glm::uvec4;
using glm::vec4;
using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

inline static GLuint createSampler() {
	GLuint sampler;
	glCreateSamplers(1u, &sampler);
	return sampler;
}

void STPSampler::STPSamplerUnbinder::operator()(STPOpenGL::STPuint unit) const {
	glBindSampler(unit, 0u);
}

void STPSampler::STPSamplerDeleter::operator()(STPOpenGL::STPuint sampler) const {
	glDeleteSamplers(1u, &sampler);
}

STPSampler::STPSampler() : Sampler(createSampler()) {

}

SuperTerrainPlus::STPOpenGL::STPuint STPSampler::operator*() const {
	return this->Sampler.get();
}

void STPSampler::filter(STPOpenGL::STPenum min, STPOpenGL::STPenum mag) {
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_MIN_FILTER, min);
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_MAG_FILTER, mag);
}

void STPSampler::wrap(STPOpenGL::STPenum s, STPOpenGL::STPenum t, STPOpenGL::STPenum r) {
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_WRAP_S, s);
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_WRAP_T, t);
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_WRAP_R, r);
}

void STPSampler::wrap(STPOpenGL::STPenum str) {
	this->wrap(str, str, str);
}

void STPSampler::borderColor(vec4 color) {
	glSamplerParameterfv(this->Sampler.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPSampler::borderColor(ivec4 color) {
	glSamplerParameterIiv(this->Sampler.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPSampler::borderColor(uvec4 color) {
	glSamplerParameterIuiv(this->Sampler.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPSampler::anisotropy(STPOpenGL::STPfloat ani) {
	glSamplerParameterf(this->Sampler.get(), GL_TEXTURE_MAX_ANISOTROPY, ani);
}

void STPSampler::compareFunction(STPOpenGL::STPint function) {
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_COMPARE_FUNC, function);
}

void STPSampler::compareMode(STPOpenGL::STPint mode) {
	glSamplerParameteri(this->Sampler.get(), GL_TEXTURE_COMPARE_MODE, mode);
}

STPSampler::STPSamplerUnitStateManager STPSampler::bindManaged(STPOpenGL::STPuint unit) const {
	glBindSampler(unit, this->Sampler.get());
	return STPSamplerUnitStateManager(unit);
}