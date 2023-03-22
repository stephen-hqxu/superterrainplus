#include <SuperRealism+/Object/STPTexture.h>
#include <SuperRealism+/Object/STPBindlessTexture.h>

//GLAD
#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

/* STPTexture */

STPTexture::STPTexture(const STPOpenGL::STPenum target) noexcept : Texture(STPSmartDeviceObject::makeGLTextureObject(target)), Target(target) {
	
}

SuperTerrainPlus::STPOpenGL::STPenum STPTexture::target() const noexcept {
	return this->Target;
}

SuperTerrainPlus::STPOpenGL::STPuint STPTexture::operator*() const noexcept {
	return this->Texture.get();
}

void STPTexture::bind(const STPOpenGL::STPuint unit) const noexcept {
	glBindTextureUnit(unit, this->Texture.get());
}

void STPTexture::bindImage(const STPOpenGL::STPuint unit, const STPOpenGL::STPint level,
	const STPOpenGL::STPboolean layered, const STPOpenGL::STPint layer, const STPOpenGL::STPenum access,
	const STPOpenGL::STPenum format) const noexcept {
	glBindImageTexture(unit, this->Texture.get(), level, layered, layer, access, format);
}

void STPTexture::unbindImage(const STPOpenGL::STPuint unit) noexcept {
	glBindImageTexture(unit, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
}

void STPTexture::unbind(const STPOpenGL::STPuint unit) noexcept {
	glBindTextureUnit(unit, 0u);
}

void STPTexture::generateMipmap() noexcept {
	glGenerateTextureMipmap(this->Texture.get());
}

void STPTexture::filter(const STPOpenGL::STPint min, const STPOpenGL::STPint mag) noexcept {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_MIN_FILTER, min);
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_MAG_FILTER, mag);
}

void STPTexture::wrap(const STPOpenGL::STPint s, const STPOpenGL::STPint t, const STPOpenGL::STPint r) noexcept {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_WRAP_S, s);
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_WRAP_T, t);
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_WRAP_R, r);
}

void STPTexture::wrap(const STPOpenGL::STPint str) noexcept {
	this->wrap(str, str, str);
}

void STPTexture::borderColor(const STPGLVector::STPfloatVec4 color) noexcept {
	glTextureParameterfv(this->Texture.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPTexture::borderColor(const STPGLVector::STPintVec4 color) noexcept {
	glTextureParameterIiv(this->Texture.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPTexture::borderColor(const STPGLVector::STPuintVec4 color) noexcept {
	glTextureParameterIuiv(this->Texture.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPTexture::anisotropy(const STPOpenGL::STPfloat ani) noexcept {
	glTextureParameterf(this->Texture.get(), GL_TEXTURE_MAX_ANISOTROPY, ani);
}

void STPTexture::compareFunction(const STPOpenGL::STPint function) noexcept {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_COMPARE_FUNC, function);
}

void STPTexture::compareMode(const STPOpenGL::STPint mode) noexcept {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_COMPARE_MODE, mode);
}

void STPTexture::textureStorage1D(const STPOpenGL::STPsizei level, const STPOpenGL::STPenum internal, const STPOpenGL::STPsizei dimension) noexcept {
	glTextureStorage1D(this->Texture.get(), level, internal, dimension);
}

void STPTexture::textureStorage2D(const STPOpenGL::STPsizei level, const STPOpenGL::STPenum internal, const STPGLVector::STPsizeiVec2 dimension) noexcept {
	glTextureStorage2D(this->Texture.get(), level, internal, dimension.x, dimension.y);
}

void STPTexture::textureStorage3D(const STPOpenGL::STPsizei level, const STPOpenGL::STPenum internal, const STPGLVector::STPsizeiVec3 dimension) noexcept {
	glTextureStorage3D(this->Texture.get(), level, internal, dimension.x, dimension.y, dimension.z);
}

void STPTexture::textureStorageMultisample2D(const STPOpenGL::STPsizei samples, const STPOpenGL::STPenum internal,
	const STPGLVector::STPsizeiVec2 dimension, const STPOpenGL::STPboolean fixed) noexcept {
	glTextureStorage2DMultisample(this->Texture.get(), samples, internal, dimension.x, dimension.y, fixed);
}

void STPTexture::textureStorageMultisample3D(const STPOpenGL::STPsizei samples, const STPOpenGL::STPenum internal,
	const STPGLVector::STPsizeiVec3 dimension, const STPOpenGL::STPboolean fixed) noexcept {
	glTextureStorage3DMultisample(this->Texture.get(), samples, internal, dimension.x, dimension.y, dimension.z, fixed);
}

void STPTexture::textureSubImage1D(const STPOpenGL::STPint level, const STPOpenGL::STPint offset, const STPOpenGL::STPsizei dimension,
	const STPOpenGL::STPenum format, const STPOpenGL::STPenum type, const void* const pixel) noexcept {
	glTextureSubImage1D(this->Texture.get(), level, offset, dimension, format, type, pixel);
}

void STPTexture::textureSubImage2D(const STPOpenGL::STPint level, const STPGLVector::STPintVec2 offset,
	const STPGLVector::STPsizeiVec2 dimension, const STPOpenGL::STPenum format, const STPOpenGL::STPenum type,
	const void* const pixel) noexcept {
	glTextureSubImage2D(this->Texture.get(), level, offset.x, offset.y, dimension.x, dimension.y, format, type, pixel);
}

void STPTexture::textureSubImage3D(const STPOpenGL::STPint level, const STPGLVector::STPintVec3 offset,
	const STPGLVector::STPsizeiVec3 dimension, const STPOpenGL::STPenum format, const STPOpenGL::STPenum type,
	const void* const pixel) noexcept {
	glTextureSubImage3D(this->Texture.get(), level, offset.x, offset.y, offset.z, dimension.x, dimension.y, dimension.z, format, type, pixel);
}

void STPTexture::getTextureImage(const STPOpenGL::STPint level, const STPOpenGL::STPenum format, const STPOpenGL::STPenum type,
	const STPOpenGL::STPsizei bufSize, void* const pixel) const noexcept {
	glGetTextureImage(this->Texture.get(), level, format, type, static_cast<GLsizei>(bufSize), pixel);
}

void STPTexture::clearTextureImage(const STPOpenGL::STPint level, const STPOpenGL::STPenum format,
	const STPOpenGL::STPenum type, const void* const data) noexcept {
	glClearTexImage(this->Texture.get(), level, format, type, data);
}

/* STPBindlessTexture */

STPBindlessTexture::STPHandle STPBindlessTexture::make(const STPTexture& texture) noexcept {
	return STPSmartDeviceObject::makeGLBindlessTextureHandle(*texture);
}

STPBindlessTexture::STPHandle STPBindlessTexture::make(const STPTexture& texture, const STPSampler& sampler) noexcept {
	return STPSmartDeviceObject::makeGLBindlessTextureHandle(*texture, *sampler);
}