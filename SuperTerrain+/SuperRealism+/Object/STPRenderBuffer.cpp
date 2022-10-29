#include <SuperRealism+/Object/STPRenderBuffer.h>

//GLAD
#include <glad/glad.h>

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

void STPRenderBuffer::renderbufferStorage(STPOpenGL::STPenum internal, STPGLVector::STPsizeiVec2 dimension) {
	glNamedRenderbufferStorage(this->RenderBuffer.get(), internal, dimension.x, dimension.y);
}

void STPRenderBuffer::renderbufferStorageMultisample(STPOpenGL::STPsizei samples, STPOpenGL::STPenum internal, STPGLVector::STPsizeiVec2 dimension) {
	glNamedRenderbufferStorageMultisample(this->RenderBuffer.get(), samples, internal, dimension.x, dimension.y);
}