#include <SuperRealism+/Object/STPTexture.h>

//GLAD
#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

using glm::ivec3;
using glm::uvec3;
using glm::ivec4;
using glm::uvec4;
using glm::vec4;
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

void STPTexture::filter(STPOpenGL::STPenum min, STPOpenGL::STPenum mag) {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_MIN_FILTER, min);
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_MAG_FILTER, mag);
}

void STPTexture::wrap(STPOpenGL::STPenum s, STPOpenGL::STPenum t, STPOpenGL::STPenum r) {
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_WRAP_S, s);
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_WRAP_T, t);
	glTextureParameteri(this->Texture.get(), GL_TEXTURE_WRAP_R, r);
}

void STPTexture::wrap(STPOpenGL::STPenum str) {
	this->wrap(str, str, str);
}

void STPTexture::borderColor(vec4 color) {
	glTextureParameterfv(this->Texture.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPTexture::borderColor(ivec4 color) {
	glTextureParameterIiv(this->Texture.get(), GL_TEXTURE_BORDER_COLOR, value_ptr(color));
}

void STPTexture::borderColor(uvec4 color) {
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

#define TEXTURE_STORAGE(DIM) \
template<> STP_REALISM_API void STPTexture::textureStorage<STPTexture::STPDimension::DIM>(STPOpenGL::STPint level, STPOpenGL::STPenum internal, uvec3 dimension)

TEXTURE_STORAGE(ONE) {
	glTextureStorage1D(this->Texture.get(), level, internal, dimension.x);
}
TEXTURE_STORAGE(TWO) {
	glTextureStorage2D(this->Texture.get(), level, internal, dimension.x, dimension.y);
}
TEXTURE_STORAGE(THREE) {
	glTextureStorage3D(this->Texture.get(), level, internal, dimension.x, dimension.y, dimension.z);
}

#define TEXTURE_STORAGE_MS(DIM) \
template<> STP_REALISM_API void STPTexture::textureStorageMultisample<STPTexture::STPDimension::DIM> \
(STPOpenGL::STPint samples, STPOpenGL::STPenum internal, uvec3 dimension, STPOpenGL::STPboolean fixed)

//TEXTURE_STORAGE_MS(ONE) can be ignored because OpenGL has no support for 1D multisampling texture

TEXTURE_STORAGE_MS(TWO) {
	glTextureStorage2DMultisample(this->Texture.get(), samples, internal, dimension.x, dimension.y, fixed);
}

TEXTURE_STORAGE_MS(THREE) {
	glTextureStorage3DMultisample(this->Texture.get(), samples, internal, dimension.x, dimension.y, dimension.z, fixed);
}

#define TEXTURE_SUBIMAGE(DIM) \
template<> STP_REALISM_API void STPTexture::textureSubImage<STPTexture::STPDimension::DIM> \
(STPOpenGL::STPint level, ivec3 offset, uvec3 dimension, STPOpenGL::STPenum format, STPOpenGL::STPenum type, const void* pixel)

TEXTURE_SUBIMAGE(ONE) {
	glTextureSubImage1D(this->Texture.get(), level, offset.x, dimension.x, format, type, pixel);
}
TEXTURE_SUBIMAGE(TWO) {
	glTextureSubImage2D(this->Texture.get(), level, offset.x, offset.y, dimension.x, dimension.y, format, type, pixel);
}
TEXTURE_SUBIMAGE(THREE) {
	glTextureSubImage3D(this->Texture.get(), level, offset.x, offset.y, offset.z, dimension.x, dimension.y, dimension.z, format, type, pixel);
}