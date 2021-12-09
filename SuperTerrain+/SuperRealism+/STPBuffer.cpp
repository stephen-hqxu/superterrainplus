#include <SuperRealism+/Object/STPBuffer.h>

//GLAD
#include <glad/glad.h>

using namespace SuperTerrainPlus::STPRealism;

inline static GLuint createBuffer() {
	GLuint buf;
	glCreateBuffers(1u, &buf);
	return buf;
}

void STPBuffer::STPBufferDeleter::operator()(STPOpenGL::STPuint buffer) const {
	glDeleteBuffers(1u, &buffer);
}

STPBuffer::STPBuffer() : Buffer(STPSmartBuffer(createBuffer())) {
	
}

SuperTerrainPlus::STPOpenGL::STPuint STPBuffer::operator*() const {
	return this->Buffer.get();
}

void STPBuffer::bind(STPOpenGL::STPenum target) const {
	glBindBuffer(target, this->Buffer.get());
}

void STPBuffer::bindBase(STPOpenGL::STPenum target, STPOpenGL::STPuint index) const {
	glBindBufferBase(target, index, this->Buffer.get());
}

void* STPBuffer::mapBuffer(STPOpenGL::STPenum access) {
	return glMapNamedBuffer(this->Buffer.get(), access);
}

void* STPBuffer::mapBufferRange(size_t length, STPOpenGL::STPintptr offset, STPOpenGL::STPbitfield access) {
	return glMapNamedBufferRange(this->Buffer.get(), offset, length, access);
}

void STPBuffer::flushMappedBufferRange(size_t length, STPOpenGL::STPintptr offset) {
	glFlushMappedNamedBufferRange(this->Buffer.get(), offset, length);
}

SuperTerrainPlus::STPOpenGL::STPboolean STPBuffer::unmapBuffer() const {
	return glUnmapNamedBuffer(this->Buffer.get());
}

void STPBuffer::unbind(STPOpenGL::STPenum target) {
	glBindBuffer(target, 0);
}

void STPBuffer::bufferStorage(size_t size, STPOpenGL::STPbitfield flag) {
	this->bufferStorageSubData(NULL, size, flag);
}

void STPBuffer::bufferSubData(const void* data, size_t size, STPOpenGL::STPintptr offset) {
	glNamedBufferSubData(this->Buffer.get(), offset, size, data);
}

void STPBuffer::bufferStorageSubData(const void* data, size_t size, STPOpenGL::STPbitfield flag) {
	glNamedBufferStorage(this->Buffer.get(), size, data, flag);
}