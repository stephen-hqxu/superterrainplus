#include <SuperRealism+/Object/STPVertexArray.h>

//Error
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>

//GLAD
#include <glad/glad.h>

using namespace SuperTerrainPlus::STPRealism;

inline static GLuint createVertexArray() noexcept {
	GLuint vao;
	glCreateVertexArrays(1u, &vao);
	return vao;
}

void STPVertexArray::STPVertexArrayDeleter::operator()(STPOpenGL::STPuint vertex_array) const noexcept {
	glDeleteVertexArrays(1u, &vertex_array);
}

STPVertexArray::STPVertexAttributeBuilder::STPVertexAttributeBuilder(STPOpenGL::STPuint vao) noexcept :
	VertexArray(vao), AttribIndex(0u), RelativeOffset(0u), BindingIndex(0u), BlockStart(0u), BlockEnd(0u) {

}

STPVertexArray::STPVertexAttributeBuilder& STPVertexArray::STPVertexAttributeBuilder::format(STPOpenGL::STPint size,
	STPOpenGL::STPenum type, STPOpenGL::STPboolean normalise, unsigned int attribSize) noexcept {
	glVertexArrayAttribFormat(this->VertexArray, this->AttribIndex, size, type, normalise, this->RelativeOffset);

	//increment counters
	this->AttribIndex++;
	this->BlockEnd++;
	this->RelativeOffset += attribSize * size;

	return *this;
}

STPVertexArray::STPVertexAttributeBuilder& STPVertexArray::STPVertexAttributeBuilder::vertexBuffer(
	const STPBuffer& buffer, STPOpenGL::STPintptr offset) noexcept {
	glVertexArrayVertexBuffer(this->VertexArray, this->BindingIndex, *buffer, offset, static_cast<GLsizei>(this->RelativeOffset));

	return *this;
}

STPVertexArray::STPVertexAttributeBuilder& STPVertexArray::STPVertexAttributeBuilder::elementBuffer(
	const STPBuffer& buffer) noexcept {
	glVertexArrayElementBuffer(this->VertexArray, *buffer);

	return *this;
}

STPVertexArray::STPVertexAttributeBuilder& STPVertexArray::STPVertexAttributeBuilder::binding() noexcept {
	for (GLuint idx = this->BlockStart; idx < this->BlockEnd; idx++) {
		glVertexArrayAttribBinding(this->VertexArray, idx, this->BindingIndex);
	}
	
	//increment
	this->BindingIndex++;
	//offset the next starting array index
	this->BlockStart = this->BlockEnd;
	//reset relative offset counter
	this->RelativeOffset = 0u;

	return *this;
}

STPVertexArray::STPVertexAttributeBuilder& STPVertexArray::STPVertexAttributeBuilder::divisor(STPOpenGL::STPint divisor) noexcept {
	glVertexArrayBindingDivisor(this->VertexArray, this->BindingIndex, divisor);
	return *this;
}

STPVertexArray::STPVertexArray() noexcept : VertexArray(createVertexArray()) {

}

void STPVertexArray::bind() const noexcept {
	glBindVertexArray(this->VertexArray.get());
}

void STPVertexArray::unbind() noexcept {
	glBindVertexArray(0);
}

SuperTerrainPlus::STPOpenGL::STPuint STPVertexArray::operator*() const noexcept {
	return this->VertexArray.get();
}

void STPVertexArray::enable(STPOpenGL::STPuint index) noexcept {
	glEnableVertexArrayAttrib(this->VertexArray.get(), index);
}

void STPVertexArray::enable(STPOpenGL::STPuint start, STPOpenGL::STPuint count) noexcept {
	for (GLuint i = 0u; i < count; i++) {
		this->enable(start + i);
	}
}

void STPVertexArray::disable(STPOpenGL::STPuint index) noexcept {
	glDisableVertexArrayAttrib(this->VertexArray.get(), index);
}

STPVertexArray::STPVertexAttributeBuilder STPVertexArray::attribute() noexcept {
	return STPVertexAttributeBuilder(this->VertexArray.get());
}