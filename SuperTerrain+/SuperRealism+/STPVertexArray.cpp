#include <SuperRealism+/Object/STPVertexArray.h>

//Error
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>

//GLAD
#include <glad/glad.h>

using namespace SuperTerrainPlus::STPRealism;

inline static GLuint createVertexArray() {
	GLuint vao;
	glCreateVertexArrays(1u, &vao);
	return vao;
}

void STPVertexArray::STPVertexArrayDeleter::operator()(STPOpenGL::STPuint vertex_array) const {
	glDeleteVertexArrays(1u, &vertex_array);
}

STPVertexArray::STPVertexAttributeBuilder::STPVertexAttributeBuilder(STPOpenGL::STPuint vao) : VertexArray(vao), 
	AttribIndex(0u), RelativeOffset(0u), BindingIndex(0u), BlockStart(0u), BlockEnd(0u) {

}

STPVertexArray::STPVertexAttributeBuilder& STPVertexArray::STPVertexAttributeBuilder::format
	(STPOpenGL::STPint size, STPOpenGL::STPenum type, STPOpenGL::STPboolean normalise, unsigned int attribSize) {
	glVertexArrayAttribFormat(this->VertexArray, this->AttribIndex, size, type, normalise, this->RelativeOffset);

	//increment counters
	this->AttribIndex++;
	this->BlockEnd++;
	this->RelativeOffset += attribSize * size;

	return *this;
}

STPVertexArray::STPVertexAttributeBuilder& STPVertexArray::STPVertexAttributeBuilder::vertexBuffer(const STPBuffer& buffer, STPOpenGL::STPintptr offset) {
	glVertexArrayVertexBuffer(this->VertexArray, this->BindingIndex, *buffer, offset, this->RelativeOffset);

	return *this;
}

STPVertexArray::STPVertexAttributeBuilder& STPVertexArray::STPVertexAttributeBuilder::elementBuffer(const STPBuffer& buffer) {
	glVertexArrayElementBuffer(this->VertexArray, *buffer);

	return *this;
}

STPVertexArray::STPVertexAttributeBuilder& STPVertexArray::STPVertexAttributeBuilder::binding() {
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

STPVertexArray::STPVertexAttributeBuilder& STPVertexArray::STPVertexAttributeBuilder::divisor(STPOpenGL::STPint divisor) {
	glVertexArrayBindingDivisor(this->VertexArray, this->BindingIndex, divisor);
	return *this;
}

STPVertexArray::STPVertexArray() : VertexArray(createVertexArray()) {

}

void STPVertexArray::bind() const {
	glBindVertexArray(this->VertexArray.get());
}

void STPVertexArray::unbind() {
	glBindVertexArray(0);
}

SuperTerrainPlus::STPOpenGL::STPuint STPVertexArray::operator*() const {
	return this->VertexArray.get();
}

void STPVertexArray::enable(STPOpenGL::STPuint index) {
	glEnableVertexArrayAttrib(this->VertexArray.get(), index);
}

void STPVertexArray::disable(STPOpenGL::STPuint index) {
	glDisableVertexArrayAttrib(this->VertexArray.get(), index);
}

STPVertexArray::STPVertexAttributeBuilder STPVertexArray::attribute() {
	return STPVertexAttributeBuilder(this->VertexArray.get());
}