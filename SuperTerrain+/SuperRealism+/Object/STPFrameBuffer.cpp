#include <SuperRealism+/Object/STPFrameBuffer.h>

#include <SuperTerrain+/Exception/STPGLError.h>

#include <sstream>

//GLAD
#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

inline static GLuint createFramebuffer() noexcept {
	GLuint fbo;
	glCreateFramebuffers(1, &fbo);
	return fbo;
}

void STPFrameBuffer::STPFrameBufferDeleter::operator()(const STPOpenGL::STPuint frame_buffer) const noexcept {
	glDeleteFramebuffers(1u, &frame_buffer);
}

STPFrameBuffer::STPFrameBuffer() noexcept : FrameBuffer(STPSmartFrameBuffer(createFramebuffer())) {

}

SuperTerrainPlus::STPOpenGL::STPuint STPFrameBuffer::operator*() const noexcept {
	return this->FrameBuffer.get();
}

void STPFrameBuffer::bind(const STPOpenGL::STPenum target) const noexcept {
	glBindFramebuffer(target, this->FrameBuffer.get());
}

void STPFrameBuffer::unbind(const STPOpenGL::STPenum target) noexcept {
	glBindFramebuffer(target, 0);
}

SuperTerrainPlus::STPOpenGL::STPenum STPFrameBuffer::status(const STPOpenGL::STPenum target) const noexcept {
	return glCheckNamedFramebufferStatus(this->FrameBuffer.get(), target);
}

void STPFrameBuffer::validate(const STPOpenGL::STPenum target) const {
	const GLenum validFlag = this->status(target);
	if (validFlag == GL_FRAMEBUFFER_COMPLETE) {
		return;
	}

	const char* flagStr = "Unknown GL framebuffer status flag";
	//convert return status to a string
	switch (validFlag) {
	case GL_FRAMEBUFFER_UNDEFINED:
		flagStr = "GL_FRAMEBUFFER_UNDEFINED";
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
		flagStr = "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
		flagStr = "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
		flagStr = "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
		flagStr = "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED:
		flagStr = "GL_FRAMEBUFFER_UNSUPPORTED";
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
		flagStr = "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
		flagStr = "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
		break;
	}

	using std::endl;
	//construct error message
	std::ostringstream err;
	err << "Framebuffer validation failed." << endl;
	err << "Status code: \n" << validFlag << "Reason: \n" << flagStr << std::endl;
	throw STPException::STPGLError(err.str().c_str());
}

void STPFrameBuffer::attach(const STPOpenGL::STPenum attachment, const STPTexture& texture, const STPOpenGL::STPint level) noexcept {
	glNamedFramebufferTexture(this->FrameBuffer.get(), attachment, *texture, level);
}

void STPFrameBuffer::attach(const STPOpenGL::STPenum attachment, const STPRenderBuffer& renderbuffer) noexcept {
	//currently OpenGL only accepts GL_RENDERBUFFER for *renderbuffer target* argument
	glNamedFramebufferRenderbuffer(this->FrameBuffer.get(), attachment, GL_RENDERBUFFER, *renderbuffer);
}

void STPFrameBuffer::detachTexture(const STPOpenGL::STPenum attachment) noexcept {
	glNamedFramebufferTexture(this->FrameBuffer.get(), attachment, 0u, 0);
}

void STPFrameBuffer::detachRenderBuffer(const STPOpenGL::STPenum attachment) noexcept {
	glNamedFramebufferRenderbuffer(this->FrameBuffer.get(), attachment, GL_RENDERBUFFER, 0u);
}

void STPFrameBuffer::drawBuffer(const STPOpenGL::STPenum buf) noexcept {
	glNamedFramebufferDrawBuffer(this->FrameBuffer.get(), buf);
}

void STPFrameBuffer::drawBuffers(const std::initializer_list<STPOpenGL::STPenum> bufs) noexcept {
	glNamedFramebufferDrawBuffers(this->FrameBuffer.get(), static_cast<GLsizei>(bufs.size()), std::data(bufs));
}

void STPFrameBuffer::readBuffer(const STPOpenGL::STPenum mode) noexcept {
	glNamedFramebufferReadBuffer(this->FrameBuffer.get(), mode);
}

void STPFrameBuffer::clearColor(const STPOpenGL::STPint drawbuffer, const STPGLVector::STPfloatVec4& colour) noexcept {
	glClearNamedFramebufferfv(this->FrameBuffer.get(), GL_COLOR, drawbuffer, value_ptr(colour));
}

void STPFrameBuffer::clearColor(const STPOpenGL::STPint drawbuffer, const STPGLVector::STPintVec4& colour) noexcept {
	glClearNamedFramebufferiv(this->FrameBuffer.get(), GL_COLOR, drawbuffer, value_ptr(colour));
}

void STPFrameBuffer::clearColor(const STPOpenGL::STPint drawbuffer, const STPGLVector::STPuintVec4& colour) noexcept {
	glClearNamedFramebufferuiv(this->FrameBuffer.get(), GL_COLOR, drawbuffer, value_ptr(colour));
}

void STPFrameBuffer::clearDepth(const STPOpenGL::STPfloat value) noexcept {
	//Following GL specification, clear depth must use *fv* version and drawbuffer must be zero.
	glClearNamedFramebufferfv(this->FrameBuffer.get(), GL_DEPTH, 0, &value);
}

void STPFrameBuffer::clearStencil(const STPOpenGL::STPint value) noexcept {
	//Like clear depth, stencil requires iv version and drawbuffer is zero.
	glClearNamedFramebufferiv(this->FrameBuffer.get(), GL_STENCIL, 0, &value);
}

void STPFrameBuffer::clearDepthStencil(const STPOpenGL::STPfloat depth, const STPOpenGL::STPint stencil) noexcept {
	//The *fi* version can only be used in this way.
	glClearNamedFramebufferfi(this->FrameBuffer.get(), GL_DEPTH_STENCIL, 0, depth, stencil);
}

void STPFrameBuffer::blitFrom(const STPFrameBuffer& readFramebuffer, const STPGLVector::STPintVec4& srcRec,
	const STPGLVector::STPintVec4& dstRec, const STPOpenGL::STPbitfield mask, const STPOpenGL::STPenum filter) noexcept {
	glBlitNamedFramebuffer(*readFramebuffer, this->FrameBuffer.get(), srcRec.x, srcRec.y, srcRec.z, srcRec.w, 
		dstRec.x, dstRec.y, dstRec.z, dstRec.w, mask, filter);
}