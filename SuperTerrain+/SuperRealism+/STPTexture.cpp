#include <SuperRealism+/Object/STPTexture.h>

//GLAD
#include <glad/glad.h>

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