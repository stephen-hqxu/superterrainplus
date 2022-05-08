#include <SuperRealism+/Object/STPBindlessBuffer.h>

//GLAD
#include <glad/glad.h>

using namespace SuperTerrainPlus::STPRealism;

void STPBindlessBuffer::STPBindlessBufferInvalidater::operator()(STPOpenGL::STPuint buffer) const {
	glMakeNamedBufferNonResidentNV(buffer);
}

inline static GLuint64EXT getBufferAddress(GLuint buffer) {
	GLuint64EXT address;
	glGetNamedBufferParameterui64vNV(buffer, GL_BUFFER_GPU_ADDRESS_NV, &address);
	return address;
}

STPBindlessBuffer::STPBindlessBuffer(const STPBuffer& buffer, STPOpenGL::STPenum access) : Buffer(*buffer), Address(getBufferAddress(this->Buffer.get())) {
	glMakeNamedBufferResidentNV(this->Buffer.get(), access);
}

SuperTerrainPlus::STPOpenGL::STPuint64 STPBindlessBuffer::operator*() const {
	return this->Address;
}