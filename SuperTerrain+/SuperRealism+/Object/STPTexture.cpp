#include <SuperRealism+/Object/STPTexture.h>

//GLAD
#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

inline static GLuint createTexture(GLenum target) {
	GLuint tex;
	glCreateTextures(target, 1u, &tex);
	return tex;
}

void STPTexture::STPTextureDeleter::operator()(STPOpenGL::STPuint texture) const {
	glDeleteTextures(1u, &texture);
}

STPTexture::STPTexture(STPOpenGL::STPenum target) : Texture(createTexture(target)), Target(target) {
	
}

SuperTerrainPlus::STPOpenGL::STPenum STPTexture::target() const {
	return this->Target;
}

SuperTerrainPlus::STPOpenGL::STPuint STPTexture::operator*() const {
	return this->Texture.get();
}

void STPTexture::bind(STPOpenGL::STPuint unit) const {
	glBindTextureUnit(unit, this->Texture.get());
}

void STPTexture::bindImage
	(STPOpenGL::STPuint unit, STPOpenGL::STPint level, STPOpenGL::STPboolean layered, STPOpenGL::STPint layer, 
		STPOpenGL::STPenum access, STPOpenGL::STPenum format) const {
	glBindImageTexture(unit, this->Texture.get(), level, layered, layer, access, format);
}

void STPTexture::unbindImage(STPOpenGL::STPuint unit) {
	glBindImageTexture(unit, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
}

void STPTexture::unbind(STPOpenGL::STPuint unit) {
	glBindTextureUnit(unit, 0u);
}

void STPTexture::generateMipmap() {
	glGenerateTextureMipmap(this->Texture.get());
}

void STPTexture::filter(STPOpenGL::STPint min, STPOpenGL::STPint mag) {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_MIN_FILTER, min);
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_MAG_FILTER, mag);
}

void STPTexture::wrap(STPOpenGL::STPint s, STPOpenGL::STPint t, STPOpenGL::STPint r) {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_WRAP_S, s);
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_WRAP_T, t);
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_WRAP_R, r);
}

void STPTexture::wrap(STPOpenGL::STPint str) {
	this->wrap(str, str, str);
}

void STPTexture::borderColor(STPGLVector::STPfloatVec4 color) {
	glTextureParameterfv(this->Texture.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPTexture::borderColor(STPGLVector::STPintVec4 color) {
	glTextureParameterIiv(this->Texture.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPTexture::borderColor(STPGLVector::STPuintVec4 color) {
	glTextureParameterIuiv(this->Texture.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPTexture::anisotropy(STPOpenGL::STPfloat ani) {
	glTextureParameterf(this->Texture.get(), GL_TEXTURE_MAX_ANISOTROPY, ani);
}

void STPTexture::compareFunction(STPOpenGL::STPint function) {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_COMPARE_FUNC, function);
}

void STPTexture::compareMode(STPOpenGL::STPint mode) {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_COMPARE_MODE, mode);
}

void STPTexture::textureStorage1D(STPOpenGL::STPsizei level, STPOpenGL::STPenum internal, STPOpenGL::STPsizei dimension) {
	glTextureStorage1D(this->Texture.get(), level, internal, dimension);
}

void STPTexture::textureStorage2D(STPOpenGL::STPsizei level, STPOpenGL::STPenum internal, STPGLVector::STPsizeiVec2 dimension) {
	glTextureStorage2D(this->Texture.get(), level, internal, dimension.x, dimension.y);
}

void STPTexture::textureStorage3D(STPOpenGL::STPsizei level, STPOpenGL::STPenum internal, STPGLVector::STPsizeiVec3 dimension) {
	glTextureStorage3D(this->Texture.get(), level, internal, dimension.x, dimension.y, dimension.z);
}

void STPTexture::textureStorageMultisample2D(STPOpenGL::STPsizei samples, STPOpenGL::STPenum internal,
	STPGLVector::STPsizeiVec2 dimension, STPOpenGL::STPboolean fixed) {
	glTextureStorage2DMultisample(this->Texture.get(), samples, internal, dimension.x, dimension.y, fixed);
}

void STPTexture::textureStorageMultisample3D(STPOpenGL::STPsizei samples, STPOpenGL::STPenum internal,
	STPGLVector::STPsizeiVec3 dimension, STPOpenGL::STPboolean fixed) {
	glTextureStorage3DMultisample(this->Texture.get(), samples, internal, dimension.x, dimension.y, dimension.z, fixed);
}

void STPTexture::textureSubImage1D(STPOpenGL::STPint level, STPOpenGL::STPint offset, STPOpenGL::STPsizei dimension,
	STPOpenGL::STPenum format, STPOpenGL::STPenum type, const void* pixel) {
	glTextureSubImage1D(this->Texture.get(), level, offset, dimension, format, type, pixel);
}

void STPTexture::textureSubImage2D(STPOpenGL::STPint level, STPGLVector::STPintVec2 offset, STPGLVector::STPsizeiVec2 dimension,
	STPOpenGL::STPenum format, STPOpenGL::STPenum type, const void* pixel) {
	glTextureSubImage2D(this->Texture.get(), level, offset.x, offset.y, dimension.x, dimension.y, format, type, pixel);
}

void STPTexture::textureSubImage3D(STPOpenGL::STPint level, STPGLVector::STPintVec3 offset,
	STPGLVector::STPsizeiVec3 dimension, STPOpenGL::STPenum format, STPOpenGL::STPenum type, const void* pixel) {
	glTextureSubImage3D(this->Texture.get(), level, offset.x, offset.y, offset.z, dimension.x, dimension.y, dimension.z, format, type, pixel);
}

void STPTexture::getTextureImage(STPOpenGL::STPint level, STPOpenGL::STPenum format, STPOpenGL::STPenum type, STPOpenGL::STPsizei bufSize, void* pixel) const {
	glGetTextureImage(this->Texture.get(), level, format, type, static_cast<GLsizei>(bufSize), pixel);
}

void STPTexture::clearTextureImage(STPOpenGL::STPint level, STPOpenGL::STPenum format, STPOpenGL::STPenum type, const void* data) {
	glClearTexImage(this->Texture.get(), level, format, type, data);
}