#include <SuperRealism+/Object/STPBuffer.h>

//GLAD
#include <glad/glad.h>

using namespace SuperTerrainPlus::STPRealism;

inline static GLuint createBuffer() noexcept {
	GLuint buf;
	glCreateBuffers(1u, &buf);
	return buf;
}

void STPBuffer::STPBufferDeleter::operator()(STPOpenGL::STPuint buffer) const noexcept {
	glDeleteBuffers(1u, &buffer);
}

STPBuffer::STPBuffer() noexcept : Buffer(STPSmartBuffer(createBuffer())) {
	
}

SuperTerrainPlus::STPOpenGL::STPuint STPBuffer::operator*() const noexcept {
	return this->Buffer.get();
}

void STPBuffer::bind(STPOpenGL::STPenum target) const noexcept {
	glBindBuffer(target, this->Buffer.get());
}

void STPBuffer::bindBase(STPOpenGL::STPenum target, STPOpenGL::STPuint index) const noexcept {
	glBindBufferBase(target, index, this->Buffer.get());
}

void STPBuffer::unbindBase(STPOpenGL::STPenum target, STPOpenGL::STPuint index) noexcept {
	glBindBufferBase(target, index, 0);
}

SuperTerrainPlus::STPOpenGL::STPuint64 STPBuffer::getAddress() const noexcept {
	GLuint64EXT address;
	glGetNamedBufferParameterui64vNV(this->Buffer.get(), GL_BUFFER_GPU_ADDRESS_NV, &address);
	return address;
}

void STPBuffer::makeResident(STPOpenGL::STPenum access) const noexcept {
	glMakeNamedBufferResidentNV(this->Buffer.get(), access);
}

void STPBuffer::makeNonResident() const noexcept {
	glMakeNamedBufferNonResidentNV(this->Buffer.get());
}

void* STPBuffer::mapBuffer(STPOpenGL::STPenum access) noexcept {
	return glMapNamedBuffer(this->Buffer.get(), access);
}

void* STPBuffer::mapBufferRange(STPOpenGL::STPintptr offset, STPOpenGL::STPsizeiptr length, STPOpenGL::STPbitfield access) noexcept {
	return glMapNamedBufferRange(this->Buffer.get(), offset, length, access);
}

void STPBuffer::flushMappedBufferRange(STPOpenGL::STPintptr offset, STPOpenGL::STPsizeiptr length) noexcept {
	glFlushMappedNamedBufferRange(this->Buffer.get(), offset, length);
}

SuperTerrainPlus::STPOpenGL::STPboolean STPBuffer::unmapBuffer() const noexcept {
	return glUnmapNamedBuffer(this->Buffer.get());
}

void STPBuffer::unbind(STPOpenGL::STPenum target) noexcept {
	glBindBuffer(target, 0);
}

void STPBuffer::bufferStorage(STPOpenGL::STPsizeiptr size, STPOpenGL::STPbitfield flag) noexcept {
	this->bufferStorageSubData(nullptr, size, flag);
}

void STPBuffer::bufferSubData(const void* data, STPOpenGL::STPsizeiptr size, STPOpenGL::STPintptr offset) noexcept {
	glNamedBufferSubData(this->Buffer.get(), offset, size, data);
}

void STPBuffer::bufferStorageSubData(const void* data, STPOpenGL::STPsizeiptr size, STPOpenGL::STPbitfield flag) noexcept {
	glNamedBufferStorage(this->Buffer.get(), size, data, flag);
}

void STPBuffer::copyBufferSubDataFrom(const STPBuffer& readBuffer, STPOpenGL::STPintptr readOffset,
	STPOpenGL::STPintptr writeOffset, STPOpenGL::STPsizeiptr size) noexcept {
	glCopyNamedBufferSubData(*readBuffer, this->Buffer.get(), readOffset, writeOffset, size);
}