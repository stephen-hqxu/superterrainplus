#include <SuperRealism+/Object/STPTexture.h>
#include <SuperRealism+/Object/STPBindlessTexture.h>

//GLAD
#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

/* STPTexture */

STPTexture::STPTexture(STPOpenGL::STPenum target) noexcept : Texture(STPSmartDeviceObject::makeGLTextureObject(target)), Target(target) {
	
}

SuperTerrainPlus::STPOpenGL::STPenum STPTexture::target() const noexcept {
	return this->Target;
}

SuperTerrainPlus::STPOpenGL::STPuint STPTexture::operator*() const noexcept {
	return this->Texture.get();
}

void STPTexture::bind(STPOpenGL::STPuint unit) const noexcept {
	glBindTextureUnit(unit, this->Texture.get());
}

void STPTexture::bindImage(STPOpenGL::STPuint unit, STPOpenGL::STPint level, STPOpenGL::STPboolean layered,
	STPOpenGL::STPint layer, STPOpenGL::STPenum access, STPOpenGL::STPenum format) const noexcept {
	glBindImageTexture(unit, this->Texture.get(), level, layered, layer, access, format);
}

void STPTexture::unbindImage(STPOpenGL::STPuint unit) noexcept {
	glBindImageTexture(unit, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
}

void STPTexture::unbind(STPOpenGL::STPuint unit) noexcept {
	glBindTextureUnit(unit, 0u);
}

void STPTexture::generateMipmap() noexcept {
	glGenerateTextureMipmap(this->Texture.get());
}

void STPTexture::filter(STPOpenGL::STPint min, STPOpenGL::STPint mag) noexcept {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_MIN_FILTER, min);
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_MAG_FILTER, mag);
}

void STPTexture::wrap(STPOpenGL::STPint s, STPOpenGL::STPint t, STPOpenGL::STPint r) noexcept {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_WRAP_S, s);
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_WRAP_T, t);
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_WRAP_R, r);
}

void STPTexture::wrap(STPOpenGL::STPint str) noexcept {
	this->wrap(str, str, str);
}

void STPTexture::borderColor(STPGLVector::STPfloatVec4 color) noexcept {
	glTextureParameterfv(this->Texture.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPTexture::borderColor(STPGLVector::STPintVec4 color) noexcept {
	glTextureParameterIiv(this->Texture.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPTexture::borderColor(STPGLVector::STPuintVec4 color) noexcept {
	glTextureParameterIuiv(this->Texture.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPTexture::anisotropy(STPOpenGL::STPfloat ani) noexcept {
	glTextureParameterf(this->Texture.get(), GL_TEXTURE_MAX_ANISOTROPY, ani);
}

void STPTexture::compareFunction(STPOpenGL::STPint function) noexcept {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_COMPARE_FUNC, function);
}

void STPTexture::compareMode(STPOpenGL::STPint mode) noexcept {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_COMPARE_MODE, mode);
}

void STPTexture::textureStorage1D(STPOpenGL::STPsizei level, STPOpenGL::STPenum internal, STPOpenGL::STPsizei dimension) noexcept {
	glTextureStorage1D(this->Texture.get(), level, internal, dimension);
}

void STPTexture::textureStorage2D(STPOpenGL::STPsizei level, STPOpenGL::STPenum internal, STPGLVector::STPsizeiVec2 dimension) noexcept {
	glTextureStorage2D(this->Texture.get(), level, internal, dimension.x, dimension.y);
}

void STPTexture::textureStorage3D(STPOpenGL::STPsizei level, STPOpenGL::STPenum internal, STPGLVector::STPsizeiVec3 dimension) noexcept {
	glTextureStorage3D(this->Texture.get(), level, internal, dimension.x, dimension.y, dimension.z);
}

void STPTexture::textureStorageMultisample2D(STPOpenGL::STPsizei samples, STPOpenGL::STPenum internal,
	STPGLVector::STPsizeiVec2 dimension, STPOpenGL::STPboolean fixed) noexcept {
	glTextureStorage2DMultisample(this->Texture.get(), samples, internal, dimension.x, dimension.y, fixed);
}

void STPTexture::textureStorageMultisample3D(STPOpenGL::STPsizei samples, STPOpenGL::STPenum internal,
	STPGLVector::STPsizeiVec3 dimension, STPOpenGL::STPboolean fixed) noexcept {
	glTextureStorage3DMultisample(this->Texture.get(), samples, internal, dimension.x, dimension.y, dimension.z, fixed);
}

void STPTexture::textureSubImage1D(STPOpenGL::STPint level, STPOpenGL::STPint offset, STPOpenGL::STPsizei dimension,
	STPOpenGL::STPenum format, STPOpenGL::STPenum type, const void* pixel) noexcept {
	glTextureSubImage1D(this->Texture.get(), level, offset, dimension, format, type, pixel);
}

void STPTexture::textureSubImage2D(STPOpenGL::STPint level, STPGLVector::STPintVec2 offset,
	STPGLVector::STPsizeiVec2 dimension, STPOpenGL::STPenum format, STPOpenGL::STPenum type,
	const void* pixel) noexcept {
	glTextureSubImage2D(this->Texture.get(), level, offset.x, offset.y, dimension.x, dimension.y, format, type, pixel);
}

void STPTexture::textureSubImage3D(STPOpenGL::STPint level, STPGLVector::STPintVec3 offset,
	STPGLVector::STPsizeiVec3 dimension, STPOpenGL::STPenum format, STPOpenGL::STPenum type,
	const void* pixel) noexcept {
	glTextureSubImage3D(this->Texture.get(), level, offset.x, offset.y, offset.z, dimension.x, dimension.y, dimension.z, format, type, pixel);
}

void STPTexture::getTextureImage(STPOpenGL::STPint level, STPOpenGL::STPenum format, STPOpenGL::STPenum type,
	STPOpenGL::STPsizei bufSize, void* pixel) const noexcept {
	glGetTextureImage(this->Texture.get(), level, format, type, static_cast<GLsizei>(bufSize), pixel);
}

void STPTexture::clearTextureImage(
	STPOpenGL::STPint level, STPOpenGL::STPenum format, STPOpenGL::STPenum type, const void* data) noexcept {
	glClearTexImage(this->Texture.get(), level, format, type, data);
}

/* STPBindlessTexture */

STPBindlessTexture::STPHandle STPBindlessTexture::make(const STPTexture& texture) noexcept {
	return STPSmartDeviceObject::makeGLBindlessTextureHandle(*texture);
}

STPBindlessTexture::STPHandle STPBindlessTexture::make(const STPTexture& texture, const STPSampler& sampler) noexcept {
	return STPSmartDeviceObject::makeGLBindlessTextureHandle(*texture, *sampler);
}