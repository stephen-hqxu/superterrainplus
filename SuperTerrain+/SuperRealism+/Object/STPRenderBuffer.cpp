#include <SuperRealism+/Object/STPRenderBuffer.h>

//GLAD
#include <glad/glad.h>

using glm::uvec2;

using namespace SuperTerrainPlus::STPRealism;

inline static GLuint createRenderBuffer() {
	GLuint rbo;
	glCreateRenderbuffers(1, &rbo);
	return rbo;
}

void STPRenderBuffer::STPRenderBufferDeleter::operator()(STPOpenGL::STPuint render_buffer) const {
	glDeleteRenderbuffers(1u, &render_buffer);
}

STPRenderBuffer::STPRenderBuffer() : RenderBuffer(STPSmartRenderBuffer(createRenderBuffer())) {

}

SuperTerrainPlus::STPOpenGL::STPuint STPRenderBuffer::operator*() const {
	return this->RenderBuffer.get();
}

void STPRenderBuffer::bind() const {
	glBindRenderbuffer(GL_RENDERBUFFER, this->RenderBuffer.get());
}

void STPRenderBuffer::unbind() {
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void STPRenderBuffer::renderbufferStorage(STPOpenGL::STPenum internal, uvec2 dimension) {
	glNamedRenderbufferStorage(this->RenderBuffer.get(), internal, dimension.x, dimension.y);
}

void STPRenderBuffer::renderbufferStorageMultisample(STPOpenGL::STPint samples, STPOpenGL::STPenum internal, uvec2 dimension) {
	glNamedRenderbufferStorageMultisample(this->RenderBuffer.get(), samples, internal, dimension.x, dimension.y);
}