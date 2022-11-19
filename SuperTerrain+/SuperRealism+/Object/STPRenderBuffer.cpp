#include <SuperRealism+/Object/STPRenderBuffer.h>

//GLAD
#include <glad/glad.h>

using namespace SuperTerrainPlus::STPRealism;

inline static GLuint createRenderBuffer() noexcept {
	GLuint rbo;
	glCreateRenderbuffers(1, &rbo);
	return rbo;
}

void STPRenderBuffer::STPRenderBufferDeleter::operator()(const STPOpenGL::STPuint render_buffer) const noexcept {
	glDeleteRenderbuffers(1u, &render_buffer);
}

STPRenderBuffer::STPRenderBuffer() noexcept : RenderBuffer(STPSmartRenderBuffer(createRenderBuffer())) {

}

SuperTerrainPlus::STPOpenGL::STPuint STPRenderBuffer::operator*() const noexcept {
	return this->RenderBuffer.get();
}

void STPRenderBuffer::bind() const noexcept {
	glBindRenderbuffer(GL_RENDERBUFFER, this->RenderBuffer.get());
}

void STPRenderBuffer::unbind() noexcept {
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void STPRenderBuffer::renderbufferStorage(const STPOpenGL::STPenum internal, const STPGLVector::STPsizeiVec2 dimension) noexcept {
	glNamedRenderbufferStorage(this->RenderBuffer.get(), internal, dimension.x, dimension.y);
}

void STPRenderBuffer::renderbufferStorageMultisample(const STPOpenGL::STPsizei samples,
	const STPOpenGL::STPenum internal, const STPGLVector::STPsizeiVec2 dimension) noexcept {
	glNamedRenderbufferStorageMultisample(this->RenderBuffer.get(), samples, internal, dimension.x, dimension.y);
}