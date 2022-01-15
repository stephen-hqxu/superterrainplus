#include <SuperRealism+/Object/STPBindlessTexture.h>

//Error
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>

//GLAD
#include <glad/glad.h>

using namespace SuperTerrainPlus::STPRealism;

void STPBindlessTexture::STPBindlessTextureInvalidater::operator()(STPOpenGL::STPuint64 handle) const {
	glMakeTextureHandleNonResidentARB(handle);
}

inline void STPBindlessTexture::enableHandle() const {
	if (!GLAD_GL_ARB_bindless_texture) {
		throw STPException::STPUnsupportedFunctionality("The current rendering context does not support ARB_bindless_texture");
	}
	glMakeTextureHandleResidentARB(this->Handle.get());
}

STPBindlessTexture::STPBindlessTexture(const STPTexture& texture) : STPBindlessTexture(*texture) {

}

STPBindlessTexture::STPBindlessTexture(STPOpenGL::STPuint texture) : Handle(glGetTextureHandleARB(texture)) {
	this->enableHandle();
}

STPBindlessTexture::STPBindlessTexture(const STPTexture& texture, const STPSampler& sampler) : STPBindlessTexture(*texture, *sampler) {

}

STPBindlessTexture::STPBindlessTexture(STPOpenGL::STPuint texture, STPOpenGL::STPuint sampler) : Handle(glGetTextureSamplerHandleARB(texture, sampler)) {
	this->enableHandle();
}

SuperTerrainPlus::STPOpenGL::STPuint64 STPBindlessTexture::operator*() const {
	return this->Handle.get();
}