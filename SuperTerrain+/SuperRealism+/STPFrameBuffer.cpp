#include <SuperRealism+/Object/STPFrameBuffer.h>

//GLAD
#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

using glm::uvec4;
using glm::ivec4;
using glm::vec4;
using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

inline static GLuint createFramebuffer() {
	GLuint fbo;
	glCreateFramebuffers(1, &fbo);
	return fbo;
}

void STPFrameBuffer::STPFrameBufferDeleter::operator()(STPOpenGL::STPuint frame_buffer) const {
	glDeleteFramebuffers(1u, &frame_buffer);
}

STPFrameBuffer::STPFrameBuffer() : FrameBuffer(STPSmartFrameBuffer(createFramebuffer())) {

}

SuperTerrainPlus::STPOpenGL::STPuint STPFrameBuffer::operator*() const {
	return this->FrameBuffer.get();
}

void STPFrameBuffer::bind(STPOpenGL::STPenum target) const {
	glBindFramebuffer(target, this->FrameBuffer.get());
}

void STPFrameBuffer::unbind(STPOpenGL::STPenum target) {
	glBindFramebuffer(target, 0);
}

SuperTerrainPlus::STPOpenGL::STPenum STPFrameBuffer::status(STPOpenGL::STPenum target) const {
	return glCheckNamedFramebufferStatus(this->FrameBuffer.get(), target);
}

void STPFrameBuffer::attach(STPOpenGL::STPenum attachment, const STPTexture& texture, STPOpenGL::STPint level) {
	glNamedFramebufferTexture(this->FrameBuffer.get(), attachment, *texture, level);
}

void STPFrameBuffer::attach(STPOpenGL::STPenum attachment, const STPRenderBuffer& renderbuffer) {
	//currently OpenGL only accepts GL_RENDERBUFFER for *renderbuffertarget* argument
	glNamedFramebufferRenderbuffer(this->FrameBuffer.get(), attachment, GL_RENDERBUFFER, *renderbuffer);
}

void STPFrameBuffer::drawBuffer(STPOpenGL::STPenum buf) {
	glNamedFramebufferDrawBuffer(this->FrameBuffer.get(), buf);
}

void STPFrameBuffer::drawBuffers(const std::vector<STPOpenGL::STPenum>& bufs) {
	glNamedFramebufferDrawBuffers(this->FrameBuffer.get(), static_cast<GLsizei>(bufs.size()), bufs.data());
}

void STPFrameBuffer::readBuffer(STPOpenGL::STPenum mode) {
	glNamedFramebufferReadBuffer(this->FrameBuffer.get(), mode);
}

#define CLEAR_COLOR(VEC) template<> STP_REALISM_API void STPFrameBuffer::clearColor<VEC>(STPOpenGL::STPint drawbuffer, const VEC& color)

CLEAR_COLOR(uvec4) {
	glClearNamedFramebufferuiv(this->FrameBuffer.get(), GL_COLOR, drawbuffer, value_ptr(color));
}

CLEAR_COLOR(ivec4) {
	glClearNamedFramebufferiv(this->FrameBuffer.get(), GL_COLOR, drawbuffer, value_ptr(color));
}

CLEAR_COLOR(vec4) {
	glClearNamedFramebufferfv(this->FrameBuffer.get(), GL_COLOR, drawbuffer, value_ptr(color));
}

void STPFrameBuffer::clearDepth(STPOpenGL::STPfloat value) {
	//Following GL specification, clear depth must use fv version and drawbuffer must be zero.
	glClearNamedFramebufferfv(this->FrameBuffer.get(), GL_DEPTH, 0, &value);
}

void STPFrameBuffer::clearStencil(STPOpenGL::STPint value) {
	//Like clear depth, stencil requires iv version and drawbuffer is zero.
	glClearNamedFramebufferiv(this->FrameBuffer.get(), GL_STENCIL, 0, &value);
}

void STPFrameBuffer::clearDepthStencil(STPOpenGL::STPfloat depth, STPOpenGL::STPint stencil) {
	//The fi version can only be used in this fasion.
	glClearNamedFramebufferfi(this->FrameBuffer.get(), GL_DEPTH_STENCIL, 0, depth, stencil);
}

void STPFrameBuffer::blitFrom(const STPFrameBuffer& readFramebuffer, const ivec4& srcRec, const ivec4& dstRec, 
	STPOpenGL::STPbitfield mask, STPOpenGL::STPenum filter) {
	glBlitNamedFramebuffer(*readFramebuffer, this->FrameBuffer.get(), srcRec.x, srcRec.y, srcRec.z, srcRec.w, 
		dstRec.x, dstRec.y, dstRec.z, dstRec.w, mask, filter);
}