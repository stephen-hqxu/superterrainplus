#include <SuperRealism+/Object/STPBindlessBuffer.h>

#include <SuperTerrain+/Exception/STPGLError.h>

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

STPBindlessBuffer::STPBindlessBuffer() : Address(0ull) {

}

STPBindlessBuffer::STPBindlessBuffer(const STPBuffer& buffer, STPOpenGL::STPenum access) : Buffer(*buffer), Address(getBufferAddress(this->Buffer.get())) {
	if (glIsNamedBufferResidentNV(this->Buffer.get())) {
		throw STPException::STPGLError("The requested buffer has already had a buffer address active");
	}
	glMakeNamedBufferResidentNV(this->Buffer.get(), access);
}

SuperTerrainPlus::STPOpenGL::STPuint64 STPBindlessBuffer::operator*() const {
	return this->Address;
}

STPBindlessBuffer::operator bool() const {
	return static_cast<bool>(this->Address);
}