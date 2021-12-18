#include <SuperRealism+/Object/STPTexture.h>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>

//GLAD
#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

using glm::ivec3;
using glm::uvec3;
using glm::vec4;
using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

/**
 * @brief Specify the dimension of a vector.
*/
enum class STPDimension : unsigned char {
	ONE = 0x01u,
	TWO = 0x02u,
	THREE = 0x03u
};

/**
 * @brief Determine the dimension of a vector.
 * Components with value of 1 is evaluated with not-enabled dimension.
 * @param dimension Vector to be evaluated.
 * @return The dimension.
*/
static STPDimension checkDimension(const uvec3& dimension) {
	if (dimension.x == 0u || dimension.y == 0u || dimension.z == 0u) {
		throw SuperTerrainPlus::STPException::STPBadNumericRange("Dimension component should not be zero");
	}

	//check the dimension of the storage
	if (dimension.y == 1u && dimension.z == 1u) {
		//1D
		return STPDimension::ONE;
	}
	if (dimension.z == 1u) {
		//2D
		return STPDimension::TWO;
	}
	//3D
	return STPDimension::THREE;
}

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
	glBindImageTexture(unit, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGB8);
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

void STPTexture::anisotropy(STPOpenGL::STPfloat ani) {
	glTextureParameterf(this->Texture.get(), GL_TEXTURE_MAX_ANISOTROPY, ani);
}

void STPTexture::textureStorage(STPOpenGL::STPint level, STPOpenGL::STPenum internal, uvec3 dimension) {
	switch (checkDimension(dimension)) {
	case STPDimension::ONE: glTextureStorage1D(this->Texture.get(), level, internal, dimension.x);
		break;
	case STPDimension::TWO: glTextureStorage2D(this->Texture.get(), level, internal, dimension.x, dimension.y);
		break;
	case STPDimension::THREE: glTextureStorage3D(this->Texture.get(), level, internal, dimension.x, dimension.y, dimension.z);
		break;
	default:
		//impossible
		break;
	}
}

void STPTexture::textureSubImage(STPOpenGL::STPint level, ivec3 offset, uvec3 dimension, STPOpenGL::STPenum format, STPOpenGL::STPenum type, const void* pixel) {
	switch (checkDimension(dimension)) {
	case STPDimension::ONE: glTextureSubImage1D(this->Texture.get(), level, offset.x, dimension.x, format, type, pixel);
		break;
	case STPDimension::TWO: glTextureSubImage2D(this->Texture.get(), level, offset.x, offset.y, dimension.x, dimension.y, format, type, pixel);
		break;
	case STPDimension::THREE: 
		glTextureSubImage3D(this->Texture.get(), level, offset.x, offset.y, offset.z, dimension.x, dimension.y, dimension.z, format, type, pixel);
		break;
	default:
		break;
	}
}